from utility import *
from setting import *
from collections import deque
from lithobench.model import *
from env import *
from model import *
from collections import namedtuple
import torch.optim as optim
from torch.nn.parallel import DataParallel
from tqdm import *


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class RLILT(ModelILT):
    def __init__(self, profiler, size=FULL_SIZE):
        super().__init__(size=size, name="RLILT")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.memory = deque([], maxlen=MEMORY_LENGTH)
        initial_img = cv2.imread(r"work/MetalSet/target/cell7.png", -1)
        initial_img = torch.tensor(initial_img, dtype=torch.float32, device=DEVICE)
        self.env = ILTEnv(profiler, initial_img, KERNEL_SIZE)

        self.policy_net = DQN_unet()
        self.target_net = DQN_unet()
        self.policy_net.to(DEVICE)
        self.target_net.to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        self.profiler = profiler
        self.select_action = self.profiler.time_func(self.select_action)
        self.optimize_model = self.profiler.time_func(self.optimize_model)
        self.pretrain = self.profiler.time_func(self.pretrain)
        self.fine_tune = self.profiler.time_func(self.fine_tune)

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._name

    def train(self, train_loader, val_loader, epochs=1, batch_size=4):
        pass


    def select_action(self, env_state, eps_threshold):
        sample = random.random()
        if ACTION_MASK is False:
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(env_state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[self.env.action_space.sample()]], device=DEVICE, dtype=torch.long)
        else:
            action_mask = get_edge(env_state.detach().clone().squeeze(0)[0])
            if sample > eps_threshold:
                with torch.no_grad():
                    q_values = self.policy_net(env_state)
                    flat_mask = action_mask.view(1, -1)
                    masked_q_values = q_values.masked_fill(flat_mask == 0, -1e9)
                    return masked_q_values.max(1)[1].view(1, 1)
            else:
                valid_actions_indices = action_mask.nonzero().squeeze(1)
                selected_action_index = random.choice(valid_actions_indices.tolist())
                return torch.tensor(
                    [[selected_action_index[1] * KERNEL_SIZE[1] + selected_action_index[0] % KERNEL_SIZE[0]]],
                    device=DEVICE, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        memory_transitions = random.sample(self.memory, BATCH_SIZE)

        states = np.stack([m.state for m in memory_transitions])
        actions = np.stack([m.action for m in memory_transitions])
        rewards = np.stack([m.reward for m in memory_transitions])

        state_batch = (
            torch.from_numpy(states)
            .to(DEVICE)
            .float()
        )
        action_batch = (
            torch.from_numpy(actions)
            .to(DEVICE)
            .long()
        )
        reward_batch = (
            torch.from_numpy(rewards)
            .to(DEVICE)
            .float()
        )

        non_final_mask = torch.tensor(
            [m.next_state is not None for m in memory_transitions],
            device=DEVICE,
            dtype=torch.bool
        )
        non_final_next_states_np = np.stack([
            m.next_state
            for m in memory_transitions
            if m.next_state is not None
        ])
        non_final_next_states = (
            torch.from_numpy(non_final_next_states_np)
            .to(DEVICE)
            .float()
        )

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        with torch.no_grad():
            if DOUBLE_DQN is True:
                # double DQN
                next_state_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)

            else:
                # DQN
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def pretrain(self, train_loader, val_loader, epochs=1, batch_size=1, logger=None, folder=None):

        best_loss = []
        last_loss = []
        epoches_reward = []
        epoches_average_min_loss = []
        interact_step = 0

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + epochs):
            logger.info(f"[Epoch {epoch} / {start_epoch + epochs}] Training")

            info = None

            epoch_min_loss = []
            epoch_last_loss = []
            progress = tqdm(train_loader)
            epoch_reward = []

            for i, (target, _) in enumerate(progress):
                self.global_step += 1
                if i > EPISODE:
                    break

                if i % UPDATE_TARGET_ITER == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                target = target.squeeze()
                target = target.to(DEVICE)

                state = self.env.reset(target_image=target)

                state = np.array([item.cpu().detach().numpy() for item in state])
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                min_loss = 1e10

                episode_reward = 0

                for timestep in range(MAX_TIMESTEP):
                    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.global_step / EPS_DECAY)
                    action = self.select_action(state, eps_threshold)

                    interact_step += 1

                    observation, reward, terminated, truncated, info = self.env.step(action.item())

                    min_loss = min((info['l2loss']), min_loss)
                    observation = np.array([item.cpu().detach().numpy() for item in observation])

                    episode_reward += reward

                    reward = torch.tensor([reward], device=DEVICE)
                    done = terminated or truncated

                    next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                    self.memory.append(
                        Transition(state.squeeze(0).detach().cpu().numpy().astype(np.uint8),
                                   action.squeeze(0).detach().cpu().numpy(),
                                   next_state.squeeze(0).detach().cpu().numpy().astype(np.uint8),
                                   reward.squeeze(0).detach().cpu().numpy()))
                    state = next_state

                    if done:
                        break
                    if interact_step % OPTIMIZE_MODEL_ITER == 0:
                        self.optimize_model()

                epoch_reward.append(episode_reward)

                epoch_min_loss.append(min_loss)
                epoch_last_loss.append(info['l2loss'])

                if i % 1000 == 0:
                    if folder:
                        self.save(os.path.join(folder, f"epoch_{epoch}_episode_{i}_model.pth"))


            average_min_loss = sum(epoch_min_loss) / len(epoch_min_loss)
            average_last_loss = sum(epoch_last_loss) / len(epoch_last_loss)

            if average_min_loss < self.best_loss:
                self.best_loss = average_min_loss
                if folder:
                    self.save(os.path.join(folder, "best_model.pth"))

            if folder:
                self.save(os.path.join(folder, f"train_epoch{epoch}.pth"))

            if logger is not None:
                logger.info(
                    f'epoch {epoch} done, average mean loss: {average_min_loss:.2f}, average last loss: {average_last_loss:.2f}')
            best_loss.append(average_min_loss)
            last_loss.append(average_last_loss)

            epoches_reward.append(epoch_reward)
            epoches_average_min_loss.append(epoch_min_loss)

        return epoches_reward, epoches_average_min_loss


    def fine_tune(self, target):
        self.memory.clear()

        target = target.squeeze()
        target = target.to(DEVICE)

        min_l2_loss = None
        best_mask = None

        if FINE_TUNE_AUGMENTATION is True:

            for epoch in tqdm(range(FINE_TUNE_EPOCH)):

                flipped_target = target.detach().clone()
                for flip in range(2):
                    with self.profiler.time_block("1.flip"):
                        flipped_target = torch.flip(flipped_target, dims=[1])
                    for rot in range(1, 5):
                        with self.profiler.time_block("2.rotate and reset"):
                            rotated_target = torch.rot90(flipped_target, k=rot, dims=[0, 1])
                            state = self.env.reset(target_image=rotated_target)
                            state = np.array([item.cpu().detach().numpy() for item in state])
                            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                        for timestep in range(MAX_TIMESTEP):
                            with self.profiler.time_block("3.load_state_dict"):
                                if timestep % UPDATE_TARGET_ITER == 0:
                                    self.target_net.load_state_dict(self.policy_net.state_dict())

                            with self.profiler.time_block("4.1 select action"):
                                eps_threshold = FINE_TUNE_EPS_END + (FINE_TUNE_EPS_START - FINE_TUNE_EPS_END) * math.exp(
                                    -1. * epoch / FINE_TUNE_EPS_DECAY)
                                action = self.select_action(state, eps_threshold)

                            with self.profiler.time_block("4.2 step"):
                                observation, reward, terminated, truncated, info = self.env.step(action.item())

                            with self.profiler.time_block("4.3 save result"):
                                target_mask = torch.clone(self.env.img_mask)

                                if flip == 1 and rot == 4:
                                    if min_l2_loss is None:
                                        min_l2_loss = info[MASK_SAVE_BY]
                                        best_mask = target_mask
                                    elif min_l2_loss > info[MASK_SAVE_BY]:
                                        min_l2_loss = info[MASK_SAVE_BY]
                                        best_mask = target_mask

                            with self.profiler.time_block("5.save memory"):
                                observation = np.array([item.cpu().detach().numpy() for item in observation])
                                reward = torch.tensor([reward], device=DEVICE)
                                done = terminated or truncated
                                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                                self.memory.append(
                                    Transition(state.squeeze(0).detach().cpu().numpy().astype(np.uint8),
                                               action.squeeze(0).detach().cpu().numpy(),
                                               next_state.squeeze(0).detach().cpu().numpy().astype(np.uint8),
                                               reward.squeeze(0).detach().cpu().numpy()))
                                state = next_state

                            with self.profiler.time_block("6.optimize model"):
                                self.optimize_model()

                            if done:
                                break


        else:
            for epoch in tqdm(range(FINE_TUNE_EPOCH)):
                tqdm.write(f"fine tune epoch: {epoch} / {FINE_TUNE_EPOCH}")

                state = self.env.reset(target_image=target)
                state = np.array([item.cpu().detach().numpy() for item in state])
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                for timestep in range(MAX_TIMESTEP):

                    with self.profiler.time_block("1.load_state_dict"):
                        if timestep % UPDATE_TARGET_ITER == 0:
                            self.target_net.load_state_dict(self.policy_net.state_dict())

                    with self.profiler.time_block("2.1 select action"):
                        eps_threshold = FINE_TUNE_EPS_END + (FINE_TUNE_EPS_START - FINE_TUNE_EPS_END) * math.exp(
                            -1. * epoch / FINE_TUNE_EPS_DECAY)
                        action = self.select_action(state, eps_threshold)
                    with self.profiler.time_block("2.2 step"):
                        observation, reward, terminated, truncated, info = self.env.step(action.item())

                    with self.profiler.time_block("2.3 save result"):
                        if min_l2_loss is None:
                            min_l2_loss = info[MASK_SAVE_BY]
                            best_mask = self.env.img_mask
                        elif min_l2_loss > info[MASK_SAVE_BY]:
                            min_l2_loss = info[MASK_SAVE_BY]
                            best_mask = self.env.img_mask

                    with self.profiler.time_block("3.save memory"):
                        observation = np.array([item.cpu().detach().numpy() for item in observation])
                        reward = torch.tensor([reward], device=DEVICE)
                        done = terminated or truncated
                        next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                        self.memory.append(
                            Transition(state.squeeze(0).detach().cpu().numpy().astype(np.uint8),
                                       action.squeeze(0).detach().cpu().numpy(),
                                       next_state.squeeze(0).detach().cpu().numpy().astype(np.uint8),
                                       reward.squeeze(0).detach().cpu().numpy()))
                        state = next_state

                    with self.profiler.time_block("4.optimize model"):
                        self.optimize_model()

                    if done:
                        break

        return best_mask

    def model_para(self):
        if torch.cuda.device_count() > 1:
            tqdm.write(f"Using {torch.cuda.device_count()} GPUs")
            self.policy_net = DataParallel(self.policy_net)
            self.target_net = DataParallel(self.target_net)

    def save(self, filenames):
        filename = filenames[0] if isinstance(filenames, list) else filenames
        save_dict = {
            'policy_net_state_dict': self._get_model_state_dict(self.policy_net),
            'target_net_state_dict': self._get_model_state_dict(self.target_net),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            # 'memory': list(self.memory)
        }
        torch.save(save_dict, filename)
        tqdm.write(f"The model has been saved to {filename}")

    def load(self, filenames, map_location=None):
        filename = filenames[0] if isinstance(filenames, list) else filenames
        if not os.path.exists(filename):
            tqdm.write(f"Error: File {filename} does not exist")
            return False

        checkpoint = torch.load(filename, map_location=map_location)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        # if 'memory' in checkpoint:
        #     self.memory = deque(checkpoint['memory'], maxlen=MEMORY_LENGTH)

        tqdm.write(f"The model has been loaded from {filename}")
        tqdm.write(f"current_epoch: {self.current_epoch}, global_step: {self.global_step}, best_loss: {self.best_loss:.6f}")
        return True

    def _get_model_state_dict(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module.state_dict()
        else:
            return model.state_dict()

    def run(self, target):

        min_l2_loss = None
        best_mask = None

        if FINE_TUNE is True:
            backup_net = DQN_unet()
            backup_net.to(DEVICE)
            if MULTIGPU and torch.cuda.device_count() > 1:
                backup_net = DataParallel(backup_net)
            backup_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.train()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

            best_mask = self.fine_tune(target)

            self.policy_net.load_state_dict(backup_net.state_dict())
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            del backup_net
            return best_mask

        for i in range(INFERENCE_EPOCH):
            target = target.squeeze()
            target = target.to(DEVICE)
            state = self.env.reset(target_image=target)
            state = np.array([item.cpu().detach().numpy() for item in state])
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            self.policy_net.eval()
            for idx in range(MAX_TIMESTEP):
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1, 1)
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                if min_l2_loss is None:
                    min_l2_loss = info[MASK_SAVE_BY]
                    best_mask = self.env.img_mask
                elif min_l2_loss > info[MASK_SAVE_BY]:
                    min_l2_loss = info[MASK_SAVE_BY]
                    best_mask = self.env.img_mask
                observation = np.array([item.cpu().detach().numpy() for item in observation])
                state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            # self.env.render()

        return best_mask
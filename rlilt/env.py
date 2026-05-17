import gymnasium as gym
from gymnasium import spaces
from utility import *
from setting import *
import pylitho.exact as litho
from collections import deque
import matplotlib.pyplot as plt


class ILTEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, profiler, target_image, kernel_size=KERNEL_SIZE):
        super(ILTEnv, self).__init__()
        self.action_space = spaces.Discrete(kernel_size[0] * kernel_size[1])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(kernel_size[0], kernel_size[1], 3),
            dtype=np.uint8
        )
        self.kernel_size = kernel_size
        self.img_size = FULL_SIZE
        self.litho = litho.LithoSim("./config/lithosimple.txt")
        self.thresh = THRESH

        self.img_target = resize_2d_tensor(target_image, self.img_size[0], self.img_size[1])
        self.target = resize_2d_tensor(self.img_target, self.kernel_size[0], self.kernel_size[1])
        self.mask = self.target

        self.img_mask = resize_2d_tensor(self.mask, self.img_size[0], self.img_size[1])
        self.printed_contour, self.printed_contour_max, self.printed_contour_min = self.litho(self.img_mask)
        self.binary_contour = torch.zeros_like(self.printed_contour)
        self.binary_contour[self.printed_contour >= self.thresh] = 1
        self.binary_contour_max = torch.zeros_like(self.printed_contour_max)
        self.binary_contour_max[self.printed_contour_max >= self.thresh] = 1
        self.binary_contour_min = torch.zeros_like(self.printed_contour_min)
        self.binary_contour_min[self.printed_contour_min >= self.thresh] = 1

        if BINARY_IMG is True:
            self.contour = resize_2d_tensor(self.binary_contour, self.kernel_size[0], self.kernel_size[1])
            with torch.no_grad():
                self.last_l2loss = func.mse_loss(self.binary_contour, self.img_target, reduction="sum").item()
                self.initial_l2loss = func.mse_loss(self.binary_contour, self.img_target, reduction="sum").item()
                self.last_pvband = torch.sum(self.binary_contour_max != self.binary_contour_min).item()
                self.initial_pvband = torch.sum(self.binary_contour_max != self.binary_contour_min).item()

        else:
            self.contour = resize_2d_tensor(self.printed_contour, self.kernel_size[0], self.kernel_size[1])
            with torch.no_grad():
                self.last_l2loss = func.mse_loss(self.printed_contour, self.img_target, reduction="sum").item()
                self.initial_l2loss = func.mse_loss(self.printed_contour, self.img_target, reduction="sum").item()
                self.last_pvband = func.mse_loss(self.printed_contour_max, self.printed_contour_min, reduction="sum").item()
                self.initial_pvband = func.mse_loss(self.printed_contour_max, self.printed_contour_min, reduction="sum").item()

        self.action_history = deque([], maxlen=RECENT_ACTION_NUM)  # 存储历史记录
        self.repeat_punishment = REPEAT_PUNISHMENT
        self.goal_reward = GOAL_REWARD
        self.timestep = 0
        self.max_timestep = MAX_TIMESTEP

        self.profiler = profiler
        self.reset = self.profiler.time_func(self.reset)
        self.step = self.profiler.time_func(self.step)


    def reset(self, target_image=None, seed=None, options=None):
        super().reset(seed=seed)
        if target_image is not None:
            self.img_target = resize_2d_tensor(target_image, self.img_size[0], self.img_size[1])
            self.target = resize_2d_tensor(self.img_target, self.kernel_size[0], self.kernel_size[1])

        self.mask = self.target

        self.img_mask = resize_2d_tensor(self.mask, self.img_size[0], self.img_size[1])
        self.printed_contour, self.printed_contour_max, self.printed_contour_min = self.litho(self.img_mask)
        self.binary_contour = torch.zeros_like(self.printed_contour)
        self.binary_contour[self.printed_contour >= self.thresh] = 1
        self.binary_contour_max = torch.zeros_like(self.printed_contour_max)
        self.binary_contour_max[self.printed_contour_max >= self.thresh] = 1
        self.binary_contour_min = torch.zeros_like(self.printed_contour_min)
        self.binary_contour_min[self.printed_contour_min >= self.thresh] = 1

        if BINARY_IMG is True:
            self.contour = resize_2d_tensor(self.binary_contour, self.kernel_size[0], self.kernel_size[1])
            with torch.no_grad():
                self.last_l2loss = func.mse_loss(self.binary_contour, self.img_target, reduction="sum").item()
                self.initial_l2loss = func.mse_loss(self.binary_contour, self.img_target, reduction="sum").item()
                self.last_pvband = torch.sum(self.binary_contour_max != self.binary_contour_min).item()
                self.initial_pvband = torch.sum(self.binary_contour_max != self.binary_contour_min).item()

        else:
            self.contour = resize_2d_tensor(self.printed_contour, self.kernel_size[0], self.kernel_size[1])
            with torch.no_grad():
                self.last_l2loss = func.mse_loss(self.printed_contour, self.img_target, reduction="sum").item()
                self.initial_l2loss = func.mse_loss(self.printed_contour, self.img_target, reduction="sum").item()
                self.last_pvband = func.mse_loss(self.printed_contour_max, self.printed_contour_min, reduction="sum").item()
                self.initial_pvband = func.mse_loss(self.printed_contour_max, self.printed_contour_min, reduction="sum").item()

        initial_observation = [self.target, self.mask, self.contour]
        self.action_history = deque([], maxlen=RECENT_ACTION_NUM)
        self.timestep = 0

        return initial_observation


    def step(self, action):
        self.timestep += 1

        x = action % self.kernel_size[0]
        y = action // self.kernel_size[1]

        self.mask = img_transfer(self.mask, x, y)

        self.img_mask = resize_2d_tensor(self.mask, self.img_size[0], self.img_size[1])
        self.printed_contour, self.printed_contour_max, self.printed_contour_min = self.litho(self.img_mask)
        self.binary_contour = torch.zeros_like(self.printed_contour)
        self.binary_contour[self.printed_contour >= self.thresh] = 1
        self.binary_contour_max = torch.zeros_like(self.printed_contour_max)
        self.binary_contour_max[self.printed_contour_max >= self.thresh] = 1
        self.binary_contour_min = torch.zeros_like(self.printed_contour_min)
        self.binary_contour_min[self.printed_contour_min >= self.thresh] = 1

        if BINARY_IMG is True:
            self.contour = resize_2d_tensor(self.binary_contour, self.kernel_size[0], self.kernel_size[1])
            l2loss = func.mse_loss(self.binary_contour, self.img_target, reduction="sum").item()
            pvband = torch.sum(self.binary_contour_max != self.binary_contour_min).item()
        else:
            self.contour = resize_2d_tensor(self.printed_contour, self.kernel_size[0], self.kernel_size[1])
            l2loss = func.mse_loss(self.printed_contour, self.img_target, reduction="sum").item()
            pvband = func.mse_loss(self.printed_contour_max, self.printed_contour_min, reduction="sum").item()

        with torch.no_grad():
            reward = L2_LOSS_REWARD_RATIO * (self.last_l2loss - l2loss) + PVBAND_REWARD_RATIO * (self.last_pvband - pvband)

            # 重复惩罚
            if len(self.action_history) > 0 and action in self.action_history:
                reward += self.repeat_punishment

            self.last_l2loss = l2loss
            self.last_pvband = pvband

        if self.last_l2loss < self.initial_l2loss * END_THRESH and self.last_pvband < self.initial_pvband * END_THRESH:
            terminated = True
            reward += self.goal_reward
        else:
            terminated = False

        truncated = False if self.timestep < self.max_timestep else True
        info = {'reward': reward, 'l2loss': l2loss, 'pvband': pvband}

        new_observation = [self.target, self.mask, self.contour]

        self.action_history.append(action)

        return new_observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(self.target.detach().cpu().numpy())
        plt.subplot(2, 3, 2)
        plt.imshow(self.mask.detach().cpu().numpy())
        plt.subplot(2, 3, 3)
        plt.imshow(self.contour.detach().cpu().numpy())
        plt.subplot(2, 3, 4)
        plt.imshow(self.img_target.detach().cpu().numpy())
        plt.subplot(2, 3, 5)
        plt.imshow(self.img_mask.detach().cpu().numpy())
        plt.subplot(2, 3, 6)
        plt.imshow(self.printed_contour.detach().cpu().numpy())

        plt.show()

    def close(self):
        pass

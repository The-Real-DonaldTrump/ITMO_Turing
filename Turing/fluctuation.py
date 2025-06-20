import argparse
import numpy as np
import pygame
import sys

# Вычисление лаплассиана с 9-точечным шаблоном
def laplacian(Z):
    return (
        -20 * Z +
        4 * (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1)) +
        1 * (np.roll(np.roll(Z, 1, 0), 1, 1) + np.roll(np.roll(Z, 1, 0), -1, 1) +
             np.roll(np.roll(Z, -1, 0), 1, 1) + np.roll(np.roll(Z, -1, 0), -1, 1))
    ) / 6.0

# Граничные условия Неймана (нулевой градиент)
def apply_neumann_boundary(U):
    U[0, :] = U[1, :]
    U[-1, :] = U[-2, :]
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gray-Scott Turing Patterns (CPU + Pygame)")
    parser.add_argument('--width',  type=int,   default=512, help='grid width')
    parser.add_argument('--height', type=int,   default=512, help='grid height')
    parser.add_argument('--Du',     type=float, default=0.16, help='diffusion rate U')
    parser.add_argument('--Dv',     type=float, default=0.08, help='diffusion rate V')
    parser.add_argument('--F',      type=float, default=0.062, help='feed rate')
    parser.add_argument('--K',      type=float, default=0.060, help='kill rate')
    parser.add_argument('--dt',     type=float, default=1.0, help='time step')
    parser.add_argument('--scale',  type=int,   default=1,   help='pixel scale')
    parser.add_argument('--fps',    type=int,   default=30,  help='frames per second')
    parser.add_argument('--steps',  type=int,   default=1,   help='simulation steps per frame')
    args = parser.parse_args()

    W, H = args.width, args.height
    Du, Dv, F, K, dt = args.Du, args.Dv, args.F, args.K, args.dt
    scale = args.scale

    # === Инициализация полей ===
    U = np.ones((H, W), dtype=np.float32)
    V = np.zeros((H, W), dtype=np.float32)

    # === Центральное пятно возмущения ===
    r = 30
    cx, cy = W // 2, H // 2
    U[cy - r:cy + r, cx - r:cx + r] = 0.00
    V[cy - r:cy + r, cx - r:cx + r] = 1.00

    # === Инициализация Pygame ===
    pygame.init()
    screen = pygame.display.set_mode((W * scale, H * scale))
    pygame.display.set_caption("Gray-Scott Model — Click to perturb")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 🖱️ Реакция на клик мыши — добавить локальное возмущение
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                mx //= scale
                my //= scale
                radius = 8  # радиус области возмущения
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        x = mx + dx
                        y = my + dy
                        if 0 <= x < W and 0 <= y < H and dx**2 + dy**2 <= radius**2:
                            U[y, x] = np.clip(U[y, x] - 0.2, 0, 1)
                            V[y, x] = np.clip(V[y, x] + 0.3, 0, 1)

        # 🔁 Модель Грея-Скотта (2-х этапный метод Эйлера)
        for _ in range(args.steps):
            Lu1 = laplacian(U)
            Lv1 = laplacian(V)
            UVV1 = U * V * V
            dU1 = Du * Lu1 - UVV1 + F * (1 - U)
            dV1 = Dv * Lv1 + UVV1 - (F + K) * V

            U_temp = U + dt * dU1
            V_temp = V + dt * dV1

            Lu2 = laplacian(U_temp)
            Lv2 = laplacian(V_temp)
            UVV2 = U_temp * V_temp * V_temp
            dU2 = Du * Lu2 - UVV2 + F * (1 - U_temp)
            dV2 = Dv * Lv2 + UVV2 - (F + K) * V_temp

            U += 0.5 * dt * (dU1 + dU2)
            V += 0.5 * dt * (dV1 + dV2)
            apply_neumann_boundary(U)
            apply_neumann_boundary(V)
            np.clip(U, 0, 1, out=U)
            np.clip(V, 0, 1, out=V)

        # === Отображение
        img = ((U - V) * 255).clip(0, 255).astype(np.uint8)
        img_rgb = np.dstack([img] * 3)
        surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
        if scale != 1:
            surf = pygame.transform.scale(surf, (W * scale, H * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sys.exit()


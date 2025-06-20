import argparse
import numpy as np
import pygame
import sys

def laplacian(Z):
    return (
        -20 * Z +
        4 * (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1)) +
        1 * (np.roll(np.roll(Z, 1, 0), 1, 1) + np.roll(np.roll(Z, 1, 0), -1, 1) +
             np.roll(np.roll(Z, -1, 0), 1, 1) + np.roll(np.roll(Z, -1, 0), -1, 1))
    ) / 6.0
def apply_neumann_boundary(U):
    U[0, :]     = U[1, :]     # верхняя граница
    U[-1, :]    = U[-2, :]    # нижняя граница
    U[:, 0]     = U[:, 1]     # левая граница
    U[:, -1]    = U[:, -2]    # правая граница
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gray-Scott Turing Patterns (CPU + Pygame)")
    parser.add_argument('--width',  type=int,   default=512, help='grid width')
    parser.add_argument('--height', type=int,   default=512, help='grid height')
    parser.add_argument('--Du',     type=float, default=0.16, help='diffusion rate U')
    parser.add_argument('--Dv',     type=float, default=0.08, help='diffusion rate V')
    parser.add_argument('--F',      type=float, default=0.035, help='feed rate')
    parser.add_argument('--K',      type=float, default=0.060, help='kill rate')
    parser.add_argument('--dt',     type=float, default=1.0, help='time step')
    parser.add_argument('--scale',  type=int,   default=1,   help='pixel scale')
    parser.add_argument('--fps',    type=int,   default=30,  help='frames per second')
    parser.add_argument('--steps', type=int, default=1, help='simulation steps per frame')
    args = parser.parse_args()

    W, H = args.width, args.height
    Du, Dv, F, K, dt = args.Du, args.Dv, args.F, args.K, args.dt
    scale = args.scale

    # === Инициализация полей ===
    U = np.ones((H, W), dtype=np.float32)
    V = np.zeros((H, W), dtype=np.float32)

    # === Центральное пятно ===
    r = 30
    cx, cy = W // 2, H // 2
    U[cy - r:cy + r, cx - r:cx + r] = 0.00
    V[cy - r:cy + r, cx - r:cx + r] = 1.00

    # === Инициализация Pygame ===
    pygame.init()
    screen = pygame.display.set_mode((W * scale, H * scale))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # === Вычисление лаплассианов ===
        Lu1 = laplacian(U)
        Lv1 = laplacian(V)
        UV2_1 = U * V * V
        dU1 = Du * Lu1 - UV2_1 + F * (1 - U)
        dV1 = Dv * Lv1 + UV2_1 - (F + K) * V

        # Предсказанный шаг (U*, V*) для оценки k2
        U_temp = U + dt * dU1
        V_temp = V + dt * dV1

        # Шаг 2: вычислить вторую производную (k2)
        Lu2 = laplacian(U_temp)
        Lv2 = laplacian(V_temp)
        UV2_2 = U_temp * V_temp * V_temp
        dU2 = Du * Lu2 - UV2_2 + F * (1 - U_temp)
        dV2 = Dv * Lv2 + UV2_2 - (F + K) * V_temp

        # Комбинация (усреднение k1 и k2)
        U += 0.5 * dt * (dU1 + dU2)
        V += 0.5 * dt * (dV1 + dV2)
        apply_neumann_boundary(U)
        apply_neumann_boundary(V)
        np.clip(U, 0, 1, out=U)
        np.clip(V, 0, 1, out=V)

        # === Отображение: по разности U - V (чёткие паттерны) ===
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

import argparse
import numpy as np
import pygame
import sys

def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
        4 * Z
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gray-Scott Turing Patterns (CPU + Pygame)")
    parser.add_argument('--width',  type=int,   default=512, help='grid width')
    parser.add_argument('--height', type=int,   default=512, help='grid height')
    parser.add_argument('--Du',     type=float, default=0.16, help='diffusion rate U')
    parser.add_argument('--Dv',     type=float, default=0.08, help='diffusion rate V')
    parser.add_argument('--F',      type=float, default=0.035, help='feed rate')
    parser.add_argument('--K',      type=float, default=0.065, help='kill rate')
    parser.add_argument('--dt',     type=float, default=1.0, help='time step')
    parser.add_argument('--scale',  type=int,   default=1,   help='pixel scale')
    parser.add_argument('--fps',    type=int,   default=30,  help='frames per second')
    args = parser.parse_args()

    W, H = args.width, args.height
    Du, Dv, F, K, dt = args.Du, args.Dv, args.F, args.K, args.dt
    scale = args.scale

    # Initialize fields U and V
    U = np.ones((H, W), dtype=np.float32)
    V = np.zeros((H, W), dtype=np.float32)
    # Small random noise in V
    V += (np.random.rand(H, W) * 0.2).astype(np.float32)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((W*scale, H*scale))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Compute laplacians
        Lu = laplacian(U)
        Lv = laplacian(V)

        # Reaction term
        UV2 = U * V * V
        dU = Du * Lu - UV2 + F * (1 - U)
        dV = Dv * Lv + UV2 - (F + K) * V

        U += dU * dt
        V += dV * dt
        # Clamp
        np.clip(U, 0, 1, out=U)
        np.clip(V, 0, 1, out=V)

        # Display V channel
        img = (V * 255).astype(np.uint8)
        # Stack to grayscale rgb
        img_rgb = np.dstack([img, img, img])
        surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0,1))
        if scale != 1:
            surf = pygame.transform.scale(surf, (W*scale, H*scale))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sys.exit()

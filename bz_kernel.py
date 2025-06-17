import argparse
import numpy as np
import pygame
import sys

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Belousov–Zhabotinsky Reaction Simulator (Pygame)')
    parser.add_argument('-W', '--width', type=int, default=512, help='Field width')
    parser.add_argument('-H', '--height', type=int, default=512, help='Field height')
    parser.add_argument('-D', '--diffusion', type=float, default=0.2, help='Diffusion rate')
    parser.add_argument('-a', type=float, default=1.0, help='Parameter a')
    parser.add_argument('-b', type=float, default=1.0, help='Parameter b')
    parser.add_argument('-c', type=float, default=1.0, help='Parameter c')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Frames per second')
    return parser.parse_args()

# Toroidal diffusion using neighbor average
def diffuse(src, rate):
    n = (
        np.roll(src, 1, 0) + np.roll(src, -1, 0) +
        np.roll(src, 1, 1) + np.roll(src, -1, 1) +
        np.roll(np.roll(src, 1, 0), 1, 1) + np.roll(np.roll(src, 1, 0), -1, 1) +
        np.roll(np.roll(src, -1, 0), 1, 1) + np.roll(np.roll(src, -1, 0), -1, 1)
    ) / 8.0
    return src * (1 - rate) + n * rate

# Reaction step with clamping
def react(a, b, c, pa, pb, pc):
    ca = a + a * (pa * b - pc * c)
    cb = b + b * (pb * c - pa * a)
    cc = c + c * (pc * a - pb * b)
    np.clip(ca, 0, 1, out=ca)
    np.clip(cb, 0, 1, out=cb)
    np.clip(cc, 0, 1, out=cc)
    return ca, cb, cc

# Main
if __name__ == '__main__':
    args = parse_args()
    W, H = args.width, args.height
    DIFF_RATE = args.diffusion
    PA, PB, PC = args.a, args.b, args.c
    FPS = args.fps

    # Initialize fields
    a = np.random.rand(H, W).astype(np.float32) * 0.1 + 0.9
    b = np.random.rand(H, W).astype(np.float32) * 0.1
    c = np.random.rand(H, W).astype(np.float32) * 0.1

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('BZ Reaction')
    clock = pygame.time.Clock()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # Update simulation
        a = diffuse(a, DIFF_RATE)
        b = diffuse(b, DIFF_RATE)
        c = diffuse(c, DIFF_RATE)
        a, b, c = react(a, b, c, PA, PB, PC)

        # Draw
        img = np.dstack((a, b, c)) * 255
        surf = pygame.surfarray.make_surface(img.astype(np.uint8).swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

import argparse
import numpy as np
import pygame
import sys

# Neighbor offsets for CA (TsukiZombina's pattern)
offsets = [(-1, -1), (-2, 0), (-1, 1),
           (1, -1),  (2,  0), (1,  1)]

# Initialize state: random 0, 1, or q
def initialize_state(h, w, q):
    rnd = np.random.rand(h, w)
    state = np.zeros((h, w), dtype=np.uint8)
    state[rnd < 0.33] = 0
    state[(rnd >= 0.33) & (rnd < 0.66)] = 1
    state[rnd >= 0.66] = q
    return state

# CA update step
def update_state(src, q, g, k1, k2):
    sum_neighbors = np.zeros_like(src, dtype=int)
    active = np.zeros_like(src, dtype=int)
    inactive = np.zeros_like(src, dtype=int)
    for dy, dx in offsets:
        nbr = np.roll(np.roll(src, dy, axis=0), dx, axis=1)
        sum_neighbors += nbr
        active += (nbr == q)
        inactive += ((nbr > 0) & (nbr != q))

    next_state = np.empty_like(src, dtype=np.uint8)
    # state == 0
    mask0 = (src == 0)
    next_state[mask0] = (active[mask0] // k1) + (inactive[mask0] // k2)
    # 0 < state < q
    mask1 = (src > 0) & (src < q)
    total_nb = active + inactive + 1
    next_state[mask1] = (sum_neighbors[mask1] // total_nb[mask1]) + g
    # state == q
    mask2 = (src == q)
    next_state[mask2] = 0
    # clamp
    np.clip(next_state, 0, 255, out=next_state)
    return next_state

# Main simulation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BZ CA Simulator (TsukiZombina)')
    parser.add_argument('-W', '--width', type=int, default=512)
    parser.add_argument('-H', '--height', type=int, default=512)
    parser.add_argument('-q', type=int, default=100, help='Max state')
    parser.add_argument('-g', type=int, default=30, help='Growth term')
    parser.add_argument('--k1', type=int, default=1)
    parser.add_argument('--k2', type=int, default=2)
    parser.add_argument('-f', '--fps', type=int, default=30)
    args = parser.parse_args()

    W, H = args.width, args.height
    Q, G, K1, K2 = args.q, args.g, args.k1, args.k2

    buf = initialize_state(H, W, Q)
    buf2 = np.zeros_like(buf)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('BZ CA Reaction')
    clock = pygame.time.Clock()
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        # Update CA
        buf2 = update_state(buf, Q, G, K1, K2)
        buf, buf2 = buf2, buf
        # Render as grayscale
        rgb = np.stack([buf]*3, axis=2)
        surf = pygame.surfarray.make_surface(rgb.astype(np.uint8).swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sys.exit()

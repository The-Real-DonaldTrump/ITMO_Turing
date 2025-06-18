import argparse
import numpy as np
import pygame
import moderngl
from pygame.locals import DOUBLEBUF, OPENGL

# GLSL shader for Belousov-Zhabotinsky step (diffusion + reaction)
VS = '''#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}'''

FS = '''#version 330
uniform sampler2D tex;
uniform float diffusion;
uniform float pa, pb, pc;
in vec2 v_uv;
out vec4 f_color;

// fetch with wrap-around
vec3 get(sampler2D t, ivec2 off, ivec2 size) {
    ivec2 coord = ivec2(v_uv * vec2(size));
    coord = (coord + off + size) % size;
    return texelFetch(t, coord, 0).rgb;
}

void main() {
    ivec2 sz = textureSize(tex, 0);
    vec3 c = texture(tex, v_uv).rgb;
    vec3 sum = vec3(0.0);
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            sum += get(tex, ivec2(dx,dy), sz);
            count++;
        }
    }
    vec3 avg = sum / float(count);
    vec3 diff = mix(c, avg, diffusion);
    // reaction
    float a = diff.r;
    float b = diff.g;
    float d = diff.b;
    float a2 = clamp(a + a*(pa*b - pc*d), 0.0, 1.0);
    float b2 = clamp(b + b*(pb*d - pa*a), 0.0, 1.0);
    float d2 = clamp(d + d*(pc*a - pb*b), 0.0, 1.0);
    f_color = vec4(a2, b2, d2, 1.0);
}'''

# Fullscreen quad
quad = np.array([ -1.0, -1.0,  1.0, -1.0,  -1.0, 1.0,
                   -1.0, 1.0,   1.0, -1.0,   1.0, 1.0 ], dtype='f4')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--width', type=int, default=512)
    parser.add_argument('-H', '--height', type=int, default=512)
    parser.add_argument('-D', '--diffusion', type=float, default=0.2)
    parser.add_argument('-a', type=float, default=1.0)
    parser.add_argument('-b', type=float, default=1.0)
    parser.add_argument('-c', type=float, default=1.0)
    parser.add_argument('-f', '--fps', type=int, default=30)
    args = parser.parse_args()

    # Pygame + OpenGL context
    pygame.init()
    pygame.display.set_mode((args.width, args.height), DOUBLEBUF|OPENGL)
    ctx = moderngl.create_context()

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')

    # Ping-pong textures
    tex1 = ctx.texture((args.width, args.height), 3, dtype='f4')
    tex2 = ctx.texture((args.width, args.height), 3, dtype='f4')
    fbo1 = ctx.framebuffer(color_attachments=[tex1])
    fbo2 = ctx.framebuffer(color_attachments=[tex2])

    # Initialize tex1 with random noise
    initial = np.dstack([np.random.rand(args.height, args.width) for _ in range(3)]).astype('f4')
    tex1.write(initial.tobytes())

    current, target = tex1, tex2
    fbo_current, fbo_target = fbo2, fbo1

    prog['diffusion'].value = args.diffusion
    prog['pa'].value = args.a
    prog['pb'].value = args.b
    prog['pc'].value = args.c

    clock = pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # Render into target FBO
        fbo_current.use()
        current.use(location=0)
        prog['tex'].value = 0
        ctx.clear()
        vao.render()

        # Swap
        current, target = target, current
        fbo_current, fbo_target = fbo_target, fbo_current

        # Display to screen
        ctx.screen.use()
        current.use(location=0)
        vao.render()
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()

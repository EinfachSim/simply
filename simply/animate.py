import sys, math, bisect
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL  import *
from OpenGL.GLU import *


# ════════════════════════════════════════════════════════════
#  Hermite interpolation
#  s0, s1 : (K, 6) state arrays — columns 0:3 pos, 3:6 vel
#  Returns interpolated positions (K, 3) at physical time t
# ════════════════════════════════════════════════════════════

def hermite_eval(t, t0, t1, s0, s1):
    p0, v0 = s0[:, :3], s0[:, 3:]
    p1, v1 = s1[:, :3], s1[:, 3:]
    dt  = t1 - t0
    tau = (t - t0) / dt
    h00 =  2*tau**3 - 3*tau**2 + 1
    h10 =    tau**3 - 2*tau**2 + tau
    h01 = -2*tau**3 + 3*tau**2
    h11 =    tau**3 -   tau**2
    return h00*p0 + h10*v0*dt + h01*p1 + h11*v1*dt


def interp_planets(planet_states, planet_times, t):
    """Return planet positions (K, 3) Hermite-interpolated at physical time t."""
    i = bisect.bisect_right(planet_times, t) - 1
    i = min(max(i, 0), len(planet_times) - 2)
    return hermite_eval(t, planet_times[i], planet_times[i+1],
                        planet_states[i], planet_states[i+1])


# ════════════════════════════════════════════════════════════
#  Entry point
#
#  pt_file   : path to traj.pt — must contain keys:
#                states  (N_pl, K, 6)
#                masses, radii, colors, names
#                tspan   [tstart, tend, dt_pl]
#
#  sc_states : (N_sc, 6) array/tensor of spacecraft states
#              on a finer uniform grid over the same [tstart, tend]
#  dt_sc     : spacecraft timestep (physical units, same as dt_pl)
# ════════════════════════════════════════════════════════════

def animate_interactive(pt_file, sc_states=None, dt_sc=None):

    import torch
    _to_np = lambda x: (x.detach().cpu().numpy().astype(np.float64)
                        if isinstance(x, torch.Tensor)
                        else np.asarray(x, np.float64))

    tensors       = torch.load(pt_file)
    planet_states = _to_np(tensors["states"])   # (N_pl, K, 6)
    names         = tensors["names"]
    radii         = np.asarray(tensors["radii"], float)
    colors        = tensors["colors"]
    masses        = tensors["masses"]

    tspan         = tensors["tspan"]
    tstart        = float(tspan[0])
    tend          = float(tspan[1])
    dt_pl         = float(tspan[2])

    N_pl, K = planet_states.shape[:2]

    # Physical time of each planet snapshot
    planet_times = np.linspace(tstart, tend, N_pl)

    # ── Spacecraft ───────────────────────────────────────────
    if sc_states is not None:
        sc_arr = _to_np(sc_states)        # (N_sc, 6)
        sc_pos = sc_arr[:, :3]            # (N_sc, 3)
        N_sc   = len(sc_pos)

        if dt_sc is not None:
            sc_times = tstart + np.arange(N_sc) * float(dt_sc)
        else:
            sc_times = np.linspace(tstart, tend, N_sc)

        print(f"Planet grid : N={N_pl}, dt={dt_pl:.4g}, t=[{tstart:.3g}, {tend:.3g}]")
        print(f"SC grid     : N={N_sc}, dt={(sc_times[1]-sc_times[0]):.4g}, t=[{sc_times[0]:.3g}, {sc_times[-1]:.3g}]")
    else:
        sc_pos   = None
        sc_times = None
        N_sc     = N_pl

    N_frames = N_sc if sc_pos is not None else N_pl

    # ── Position helpers ─────────────────────────────────────

    def sc_position(frame):
        """Direct lookup — sc grid is the master clock."""
        return sc_pos[frame]

    def planet_position(frame):
        """Hermite-interpolated planet positions at the physical time of sc frame."""
        t = sc_times[frame] if sc_times is not None else planet_times[frame]
        return interp_planets(planet_states, planet_times, t)  # (K, 3)

    def planet_trail(frame, n_trail):
        """Planet positions (trail_len, K, 3) at the sc-grid times of the trail window."""
        f0 = max(0, frame - n_trail)
        ts = sc_times[f0:frame + 1] if sc_times is not None else planet_times[f0:frame + 1]
        return np.stack([interp_planets(planet_states, planet_times, t) for t in ts])

    # ── Visual setup ─────────────────────────────────────────
    try:
        import matplotlib.colors as _mc
        colors = [_mc.to_rgb(c) if isinstance(c, str) else tuple(c) for c in colors]
    except ImportError:
        colors = [tuple(c) for c in colors]

    all_pos = planet_states[:, :, :3].reshape(-1, 3)
    span    = float(max((all_pos.max(0) - all_pos.min(0)).max(), 1e-6))
    ctr     = (all_pos.max(0) + all_pos.min(0)) / 2.0
    vis_r   = radii.copy()

    TRAIL         = max(200, N_frames // 10)
    SPEED_MIN     = 0.5
    SPEED_MAX     = float(max(N_frames // 10, 2))
    MIN_SCREEN_PX = 6.0
    SC_COL        = (0.9, 0.9, 0.2)
    SC_R          = float(radii.min()) * 0.05


    # ════════════════════════════════════════════════════════════
    #  Camera
    # ════════════════════════════════════════════════════════════
    class Camera:
        def __init__(self):
            self.target = ctr.copy()
            self.dist   = span * 2.5
            self.yaw    = -30.0
            self.pitch  =  25.0

        def eye(self):
            yr, pr = math.radians(self.yaw), math.radians(self.pitch)
            return self.target + self.dist * np.array([
                math.cos(pr) * math.sin(yr),
                math.sin(pr),
                math.cos(pr) * math.cos(yr),
            ])

        def look(self):
            e = self.eye()
            glLoadIdentity()
            gluLookAt(e[0], e[1], e[2],
                      self.target[0], self.target[1], self.target[2],
                      0, 1, 0)

        def orbit(self, dx, dy):
            self.yaw   = (self.yaw - dx * 0.4) % 360
            self.pitch = max(-89, min(89, self.pitch + dy * 0.4))

        def pan(self, dx, dy):
            yr, pr = math.radians(self.yaw), math.radians(self.pitch)
            right  = np.array([ math.cos(yr), 0, -math.sin(yr)])
            fwd    = np.array([-math.cos(pr) * math.sin(yr),
                                math.sin(pr),
                               -math.cos(pr) * math.cos(yr)])
            up     = np.cross(right, -fwd)
            s      = self.dist * 0.0012
            self.target += -right * dx * s + up * dy * s

        def zoom(self, clicks):
            self.dist = max(span * 1e-6, min(span * 50, self.dist * (0.88 ** clicks)))

        def reset(self):
            self.__init__()

    cam = Camera()


    # ════════════════════════════════════════════════════════════
    #  Sphere geometry
    # ════════════════════════════════════════════════════════════
    def _make_sphere(stacks=18, slices=28):
        v = []
        for i in range(stacks):
            a0 = math.pi * (-0.5 + i / stacks)
            a1 = math.pi * (-0.5 + (i + 1) / stacks)
            z0, zr0 = math.sin(a0), math.cos(a0)
            z1, zr1 = math.sin(a1), math.cos(a1)
            for j in range(slices):
                b0 = 2 * math.pi * j / slices
                b1 = 2 * math.pi * (j + 1) / slices
                v += [(zr0*math.cos(b0), z0, zr0*math.sin(b0)),
                      (zr1*math.cos(b0), z1, zr1*math.sin(b0)),
                      (zr1*math.cos(b1), z1, zr1*math.sin(b1)),
                      (zr0*math.cos(b1), z0, zr0*math.sin(b1))]
        return np.array(v, np.float32)

    _SPH = _make_sphere()

    def draw_sphere(x, y, z, r, col, alpha=1.0):
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(r, r, r)
        glColor4f(*col, alpha)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, _SPH)
        glDrawArrays(GL_QUADS, 0, len(_SPH))
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

    def draw_glow(x, y, z, r, col, scale, alpha):
        glDepthMask(GL_FALSE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        draw_sphere(x, y, z, r * scale, col, alpha)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_TRUE)

    def draw_lock_ring(x, y, z, r):
        glPushMatrix()
        glTranslatef(x, y, z)
        glLineWidth(1.8)
        glColor4f(1.0, 0.85, 0.2, 0.9)
        glBegin(GL_LINE_LOOP)
        for s in range(56):
            a = 2 * math.pi * s / 56
            glVertex3f(math.cos(a) * r, 0, math.sin(a) * r)
        glEnd()
        glLineWidth(1.0)
        glPopMatrix()


    # ════════════════════════════════════════════════════════════
    #  Screen-space radius
    # ════════════════════════════════════════════════════════════
    def get_screen_radius(x, y, z, world_r):
        mv  = (GLdouble * 16)(); glGetDoublev(GL_MODELVIEW_MATRIX,  mv)
        prj = (GLdouble * 16)(); glGetDoublev(GL_PROJECTION_MATRIX, prj)
        vp  = (GLint   *  4)(); glGetIntegerv(GL_VIEWPORT,           vp)
        sx, sy, _ = gluProject(x, y, z, mv, prj, vp)
        ex, ey, _ = gluProject(x + world_r, y, z, mv, prj, vp)
        screen_r  = math.hypot(ex - sx, ey - sy)
        if screen_r < 0.01:
            return False, None
        if screen_r >= MIN_SCREEN_PX:
            return True, world_r
        return False, world_r * (MIN_SCREEN_PX / screen_r)


    # ════════════════════════════════════════════════════════════
    #  Stars
    # ════════════════════════════════════════════════════════════
    rng = np.random.default_rng(7)
    _ST = rng.standard_normal((2000, 3)).astype(np.float32)
    _ST /= np.linalg.norm(_ST, axis=1, keepdims=True)
    _ST *= span * 80

    def draw_stars():
        glDepthMask(GL_FALSE)
        glPointSize(1.3)
        glColor4f(0.85, 0.88, 1.0, 0.55)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, _ST)
        glDrawArrays(GL_POINTS, 0, len(_ST))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDepthMask(GL_TRUE)


    # ════════════════════════════════════════════════════════════
    #  Trail
    # ════════════════════════════════════════════════════════════
    def draw_trail(pts, col):
        M = len(pts)
        if M < 2:
            return
        glLineWidth(1.1)
        glBegin(GL_LINE_STRIP)
        for i, p in enumerate(pts):
            glColor4f(*col, (i / (M - 1)) * 0.6)
            glVertex3f(*p)
        glEnd()


    # ════════════════════════════════════════════════════════════
    #  Picking
    # ════════════════════════════════════════════════════════════
    def pick(mx, my, H, pl_pos, sc_p):
        mv  = (GLdouble * 16)(); glGetDoublev(GL_MODELVIEW_MATRIX,  mv)
        prj = (GLdouble * 16)(); glGetDoublev(GL_PROJECTION_MATRIX, prj)
        vp  = (GLint   *  4)(); glGetIntegerv(GL_VIEWPORT,           vp)
        best, bestd = None, 1e9

        for k in range(K):
            p = pl_pos[k]
            sx, sy, _ = gluProject(p[0], p[1], p[2], mv, prj, vp)
            sy = H - sy
            ex, ey, _ = gluProject(p[0] + vis_r[k], p[1], p[2], mv, prj, vp)
            sr = max(abs(ex - sx), MIN_SCREEN_PX)
            d  = math.hypot(mx - sx, my - sy)
            if d < sr * 1.4 and d < bestd:
                bestd, best = d, k

        if sc_p is not None:
            sx, sy, _ = gluProject(sc_p[0], sc_p[1], sc_p[2], mv, prj, vp)
            sy = H - sy
            ex, ey, _ = gluProject(sc_p[0] + SC_R, sc_p[1], sc_p[2], mv, prj, vp)
            sr = max(abs(ex - sx), MIN_SCREEN_PX)
            d  = math.hypot(mx - sx, my - sy)
            if d < sr * 1.4 and d < bestd:
                bestd, best = d, 'sc'

        return best


    # ════════════════════════════════════════════════════════════
    #  HUD
    # ════════════════════════════════════════════════════════════
    pygame.font.init()
    try:
        Flg = pygame.font.SysFont("monospace", 16, bold=True)
        Fsm = pygame.font.SysFont("monospace", 13)
    except Exception:
        Flg = pygame.font.Font(None, 22)
        Fsm = pygame.font.Font(None, 17)

    def _txt(surf, text, pos, font, col=(200, 200, 230)):
        surf.blit(font.render(text, True, col), pos)

    def slider_rect(H):
        return (14, H - 58, 260, 6)

    def speed_from_mouse(mx, H):
        SBX, _, SBW, _ = slider_rect(H)
        t = max(0.0, min(1.0, (mx - SBX) / SBW))
        return math.exp(math.log(SPEED_MIN) + t * (math.log(SPEED_MAX) - math.log(SPEED_MIN)))

    def on_slider(mx, my, H):
        SBX, SBY, SBW, SBH = slider_rect(H)
        return SBX <= mx <= SBX + SBW and abs(my - (SBY + SBH // 2)) <= 14

    def draw_hud(surf, W, H, frame, playing, locked_name, speed, dragging_slider):
        surf.fill((0, 0, 0, 0))

        t_phys = sc_times[frame] if sc_times is not None else planet_times[frame]
        _txt(surf, f"SC FRAME {frame:06d}/{N_frames-1}  t={t_phys:.3f}", (14, 10), Flg, (140, 140, 220))
        _txt(surf, "▶ PLAYING" if playing else "⏸ PAUSED", (14, 32), Fsm,
             (80, 220, 120) if playing else (220, 120, 80))
        _txt(surf,
             f"LOCKED  {locked_name}" if locked_name else "INERTIAL FRAME",
             (14, 50), Fsm,
             (255, 195, 60) if locked_name else (90, 110, 170))

        SBX, SBY, SBW, SBH = slider_rect(H)
        pygame.draw.rect(surf, (35, 35, 70), (SBX, SBY, SBW, SBH), border_radius=3)
        t = (math.log(speed) - math.log(SPEED_MIN)) / (math.log(SPEED_MAX) - math.log(SPEED_MIN))
        t = max(0.0, min(1.0, t))
        pygame.draw.rect(surf, (70, 95, 190), (SBX, SBY, int(SBW * t), SBH), border_radius=3)
        hx   = SBX + int(SBW * t)
        hcol = (200, 220, 255) if dragging_slider else (120, 145, 220)
        pygame.draw.circle(surf, hcol, (hx, SBY + SBH // 2), 8)
        _txt(surf, "SPEED", (SBX, SBY - 17), Fsm, (80, 80, 130))
        _txt(surf, f"{speed:.1f} sc-frames/frame", (SBX + SBW + 10, SBY - 5), Fsm, (130, 130, 200))

        for i, h in enumerate(reversed([
            "RIGHT-DRAG orbit   MID-DRAG pan   SCROLL zoom",
            "LEFT-CLICK planet/sc=lock (again=release)   R reset",
            "SPACE play/pause   ← → step 1 frame   Q quit",
        ])):
            _txt(surf, h, (14, H - 78 - i * 17), Fsm, (55, 55, 95))

        for k in range(K):
            c = tuple(int(x * 255) for x in colors[k])
            pygame.draw.circle(surf, c, (W - 130, 14 + k * 19), 5)
            _txt(surf, names[k], (W - 120, 7 + k * 19), Fsm, c)

        if sc_pos is not None:
            pygame.draw.circle(surf, (230, 230, 50), (W - 130, 14 + K * 19), 5)
            _txt(surf, "spacecraft", (W - 120, 7 + K * 19), Fsm, (230, 230, 50))


    # ════════════════════════════════════════════════════════════
    #  Window / GL init
    # ════════════════════════════════════════════════════════════
    W, H = 1280, 800
    pygame.init()
    screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("N-Body Simulator")
    hud = pygame.Surface((W, H), pygame.SRCALPHA)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glDisable(GL_LIGHTING)
    glShadeModel(GL_FLAT)
    glClearColor(0.03, 0.03, 0.09, 1.0)

    def set_proj(w, h):
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(50, w / max(h, 1), max(span * 0.0001, 1e-6), span * 300)
        glMatrixMode(GL_MODELVIEW)
    set_proj(W, H)


    # ════════════════════════════════════════════════════════════
    #  Loop state
    # ════════════════════════════════════════════════════════════
    frame           = 0       # integer index into sc grid (master clock)
    playing         = False
    speed           = 1.0     # sc-frames to advance per display frame
    frame_acc       = 0.0     # sub-frame accumulator
    locked          = None    # int planet index, 'sc', or None
    drag            = None
    last_m          = (0, 0)
    dragging_slider = False
    clock           = pygame.time.Clock()


    # ════════════════════════════════════════════════════════════
    #  Main loop
    # ════════════════════════════════════════════════════════════
    while True:
        clock.tick(60)

        # ── Events ───────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit(); sys.exit()

            elif ev.type == VIDEORESIZE:
                W, H   = ev.w, ev.h
                screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | RESIZABLE)
                hud    = pygame.Surface((W, H), pygame.SRCALPHA)
                glViewport(0, 0, W, H); set_proj(W, H)

            elif ev.type == KEYDOWN:
                if ev.key in (K_q, K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif ev.key == K_SPACE:
                    playing = not playing
                elif ev.key == K_RIGHT:
                    playing = False
                    frame   = min(frame + 1, N_frames - 1)
                elif ev.key == K_LEFT:
                    playing = False
                    frame   = max(frame - 1, 0)
                elif ev.key == K_r:
                    cam.reset(); locked = None

            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if on_slider(ev.pos[0], ev.pos[1], H):
                        dragging_slider = True
                        speed = speed_from_mouse(ev.pos[0], H)
                    else:
                        k = pick(ev.pos[0], ev.pos[1], H, pl_pos, sc_p)
                        locked = None if locked == k else k
                elif ev.button == 3:
                    drag = 'orbit'; last_m = ev.pos
                elif ev.button == 2:
                    drag = 'pan';   last_m = ev.pos
                elif ev.button == 4:
                    cam.zoom(-1)
                elif ev.button == 5:
                    cam.zoom( 1)

            elif ev.type == MOUSEBUTTONUP:
                if ev.button == 1:
                    dragging_slider = False
                if (ev.button == 3 and drag == 'orbit') or \
                   (ev.button == 2 and drag == 'pan'):
                    drag = None

            elif ev.type == MOUSEMOTION:
                if dragging_slider:
                    speed = speed_from_mouse(ev.pos[0], H)
                elif drag is not None:
                    dx = ev.pos[0] - last_m[0]; dy = ev.pos[1] - last_m[1]
                    if drag == 'orbit': cam.orbit(dx, dy)
                    else:               cam.pan(dx, dy)
                    last_m = ev.pos

        # ── Advance ──────────────────────────────────────────
        if playing:
            frame_acc += speed
            steps      = int(frame_acc)
            frame_acc -= steps
            frame      = (frame + steps) % N_frames

        # ── Positions (computed after advance, match the frame being drawn)
        pl_pos = planet_position(frame)                              # (K, 3)
        sc_p   = sc_position(frame) if sc_pos is not None else None  # (3,)

        # ── Camera lock ───────────────────────────────────────
        if locked == 'sc' and sc_p is not None:
            cam.target = sc_p.copy()
        elif isinstance(locked, int):
            cam.target = pl_pos[locked].copy()

        # ── Render ───────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cam.look()
        draw_stars()

        # ── Planets ──────────────────────────────────────────
        trail_pl = planet_trail(frame, TRAIL)   # (trail_len, K, 3)

        for k in range(K):
            col = colors[k]
            draw_trail(trail_pl[:, k, :], col)

            p = pl_pos[k]
            use_sphere, draw_r = get_screen_radius(p[0], p[1], p[2], vis_r[k])
            if draw_r is None:
                continue

            body_alpha = 1.0  if use_sphere else 0.75
            glow_scale = 2.6  if use_sphere else 4.0
            glow_alpha = 0.10 if use_sphere else 0.18
            draw_sphere(p[0], p[1], p[2], draw_r, col, body_alpha)
            draw_glow(  p[0], p[1], p[2], draw_r, col, glow_scale, glow_alpha)

            if locked == k:
                draw_lock_ring(p[0], p[1], p[2], draw_r * 1.8)

        # ── Spacecraft ───────────────────────────────────────
        if sc_p is not None:
            draw_trail(sc_pos[max(0, frame - TRAIL): frame + 1], SC_COL)

            use_sphere, draw_r = get_screen_radius(sc_p[0], sc_p[1], sc_p[2], SC_R)
            if draw_r is not None:
                draw_sphere(sc_p[0], sc_p[1], sc_p[2], draw_r, SC_COL, 1.0)
                draw_glow(  sc_p[0], sc_p[1], sc_p[2], draw_r, SC_COL, 4.0, 0.20)
                if locked == 'sc':
                    draw_lock_ring(sc_p[0], sc_p[1], sc_p[2], draw_r * 1.8)

        # ── HUD ──────────────────────────────────────────────
        lname = ("spacecraft"  if locked == 'sc'
                 else names[locked] if isinstance(locked, int)
                 else None)
        draw_hud(hud, W, H, frame, playing, lname, speed, dragging_slider)
        raw = pygame.image.tostring(hud, "RGBA", True)
        glWindowPos2i(0, 0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, raw)

        pygame.display.flip()
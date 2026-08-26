"""
Planet Animator — 3D (PyOpenGL + Pygame)
=========================================
  states : torch.Tensor or np.ndarray  (N, K, 6)  [x,y,z,vx,vy,vz]
  radii  : array (K,)
  names  : list[str] (K,)
  colors : list of RGB tuples (K,)

CONTROLS
  Right-drag   Orbit
  Middle-drag  Pan
  Scroll       Zoom
  Left-click   Lock/unlock camera on a planet
  Space        Pause / play
  ← →          Step one frame (paused)
  R            Reset camera
  Q / Esc      Quit
"""

import sys, math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL  import *
from OpenGL.GLU import *

try:
    import torch
    _to_np = lambda x: x.detach().cpu().numpy().astype(np.float64) if isinstance(x, torch.Tensor) else np.asarray(x, np.float64)
except ImportError:
    _to_np = lambda x: np.asarray(x, np.float64)


from simply.universe import Universe

uni = Universe("universe.yaml")

states = uni.simulate(0, 10, 0.001)
radii  = [body.radius for body in uni.bodies]
names  = [body.name   for body in uni.bodies]
colors = [body.color  for body in uni.bodies]
masses = [body.mass for body in uni.bodies]

import torch

def total_energy(state, masses, G=1.0):
    pos  = state[:, :3]
    vel  = state[:, 3:]
    masses = torch.tensor(masses, dtype=torch.float64)
    KE = 0.5 * (masses * (vel**2).sum(dim=1)).sum()
    PE = 0.0
    K  = pos.shape[0]
    for i in range(K):
        for j in range(i+1, K):
            r   = (pos[i] - pos[j]).norm()
            PE -= G * masses[i] * masses[j] / r
    return (KE + PE).item()

print("E0 =",    total_energy(states[0],    masses))
print("E1 =",    total_energy(states[100],  masses))
print("E_end =", total_energy(states[-1],   masses))


pos  = _to_np(states)[:,:,:3].copy()   # (N,K,3)  world-space, never modified
N, K = pos.shape[:2]
radii = np.asarray(radii, float)

try:
    import matplotlib.colors as _mc
    colors = [_mc.to_rgb(c) if isinstance(c,str) else tuple(c) for c in colors]
except ImportError:
    colors = [tuple(c) for c in colors]

span = float(max((pos.reshape(-1,3).max(0)-pos.reshape(-1,3).min(0)).max(), 1e-6))
ctr  = (pos.reshape(-1,3).max(0)+pos.reshape(-1,3).min(0))/2.0

# True physical radii — screen-space minimum is handled at draw time
vis_r = radii.copy()

TRAIL = max(50, N//25)
SPEED_MIN, SPEED_MAX = 0.05, 30.0
MIN_SCREEN_PX = 6.0   # minimum rendered size in pixels


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
            math.cos(pr)*math.sin(yr),
            math.sin(pr),
            math.cos(pr)*math.cos(yr)
        ])

    def look(self):
        e = self.eye()
        glLoadIdentity()
        gluLookAt(e[0],e[1],e[2],
                  self.target[0],self.target[1],self.target[2],
                  0,1,0)

    def orbit(self, dx, dy):
        self.yaw   = (self.yaw - dx*0.4) % 360
        self.pitch = max(-89, min(89, self.pitch + dy*0.4))

    def pan(self, dx, dy):
        yr, pr = math.radians(self.yaw), math.radians(self.pitch)
        right = np.array([ math.cos(yr), 0, -math.sin(yr)])
        fwd   = np.array([-math.cos(pr)*math.sin(yr),
                           math.sin(pr),
                          -math.cos(pr)*math.cos(yr)])
        up    = np.cross(right, -fwd)
        s     = self.dist * 0.0012
        self.target += -right*dx*s + up*dy*s

    def zoom(self, clicks):
        self.dist = max(span*0.000001, min(span*50, self.dist * (0.88**clicks)))

    def reset(self):
        self.__init__()

cam = Camera()


# ════════════════════════════════════════════════════════════
#  Sphere geometry
# ════════════════════════════════════════════════════════════
def _make_sphere(stacks=18, slices=28):
    v = []
    for i in range(stacks):
        a0,a1 = math.pi*(-0.5+i/stacks), math.pi*(-0.5+(i+1)/stacks)
        z0,zr0 = math.sin(a0),math.cos(a0)
        z1,zr1 = math.sin(a1),math.cos(a1)
        for j in range(slices):
            b0,b1 = 2*math.pi*j/slices, 2*math.pi*(j+1)/slices
            v += [(zr0*math.cos(b0),z0,zr0*math.sin(b0)),
                  (zr1*math.cos(b0),z1,zr1*math.sin(b0)),
                  (zr1*math.cos(b1),z1,zr1*math.sin(b1)),
                  (zr0*math.cos(b1),z0,zr0*math.sin(b1))]
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


# ════════════════════════════════════════════════════════════
#  Screen-space radius helper
# ════════════════════════════════════════════════════════════
def get_screen_radius(x, y, z, world_r):
    """Returns (use_sphere, draw_r).
    use_sphere: True  → body is big enough to render as real sphere
                False → body is tiny, draw_r is inflated to MIN_SCREEN_PX
    draw_r: world-space radius to actually draw (may be inflated)
    """
    mv  = (GLdouble*16)(); glGetDoublev(GL_MODELVIEW_MATRIX,  mv)
    prj = (GLdouble*16)(); glGetDoublev(GL_PROJECTION_MATRIX, prj)
    vp  = (GLint*4)();     glGetIntegerv(GL_VIEWPORT,          vp)

    sx, sy, _ = gluProject(x, y, z, mv, prj, vp)
    ex, ey, _ = gluProject(x + world_r, y, z, mv, prj, vp)
    screen_r  = math.hypot(ex-sx, ey-sy)   # pixels

    if screen_r < 0.01:
        return False, None                  # behind camera / degenerate

    if screen_r >= MIN_SCREEN_PX:
        return True, world_r               # real size is already big enough

    # inflate so it always covers MIN_SCREEN_PX pixels
    return False, world_r * (MIN_SCREEN_PX / screen_r)


# ════════════════════════════════════════════════════════════
#  Stars
# ════════════════════════════════════════════════════════════
rng  = np.random.default_rng(7)
_ST  = rng.standard_normal((2000,3)).astype(np.float32)
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
    if M < 2: return
    glLineWidth(1.1)
    glBegin(GL_LINE_STRIP)
    for i, p in enumerate(pts):
        a = (i/(M-1)) * 0.6
        glColor4f(*col, a)
        glVertex3f(*p)
    glEnd()


# ════════════════════════════════════════════════════════════
#  Picking
# ════════════════════════════════════════════════════════════
def pick(mx, my, H, frame):
    mv  = (GLdouble*16)(); glGetDoublev(GL_MODELVIEW_MATRIX,  mv)
    prj = (GLdouble*16)(); glGetDoublev(GL_PROJECTION_MATRIX, prj)
    vp  = (GLint*4)();     glGetIntegerv(GL_VIEWPORT,          vp)
    best, bestd = None, 1e9
    for k in range(K):
        p = pos[frame, k]
        sx, sy, _ = gluProject(p[0],p[1],p[2], mv,prj,vp)
        sy = H - sy
        ex, ey, _ = gluProject(p[0]+vis_r[k],p[1],p[2], mv,prj,vp)
        sr = max(abs(ex-sx), MIN_SCREEN_PX)   # at least MIN_SCREEN_PX clickable
        d  = math.hypot(mx-sx, my-sy)
        if d < sr*1.4 and d < bestd:
            bestd, best = d, k
    return best


# ════════════════════════════════════════════════════════════
#  HUD fonts
# ════════════════════════════════════════════════════════════
pygame.font.init()
try:
    Flg = pygame.font.SysFont("monospace", 16, bold=True)
    Fsm = pygame.font.SysFont("monospace", 13)
except:
    Flg = pygame.font.Font(None, 22)
    Fsm = pygame.font.Font(None, 17)

def _txt(surf, text, pos, font, col=(200,200,230)):
    surf.blit(font.render(text, True, col), pos)


# ════════════════════════════════════════════════════════════
#  Slider helpers
# ════════════════════════════════════════════════════════════
def slider_rect(H):
    return (14, H-58, 260, 6)

def speed_from_mouse(mx, H):
    SBX, _, SBW, _ = slider_rect(H)
    t = max(0.0, min(1.0, (mx-SBX)/SBW))
    return math.exp(math.log(SPEED_MIN) + t*(math.log(SPEED_MAX)-math.log(SPEED_MIN)))

def on_slider(mx, my, H):
    SBX, SBY, SBW, SBH = slider_rect(H)
    return SBX <= mx <= SBX+SBW and abs(my-(SBY+SBH//2)) <= 14


# ════════════════════════════════════════════════════════════
#  HUD draw
# ════════════════════════════════════════════════════════════
def draw_hud(surf, W, H, frame, playing, locked, speed, dragging_slider):
    surf.fill((0,0,0,0))

    _txt(surf, f"FRAME {frame:05d}/{N-1}", (14,10), Flg, (140,140,220))
    _txt(surf, ("▶ PLAYING" if playing else "⏸ PAUSED"), (14,32), Fsm,
         (80,220,120) if playing else (220,120,80))
    lname = names[locked] if locked is not None else None
    _txt(surf, f"LOCKED  {lname}" if lname else "INERTIAL FRAME",
         (14,50), Fsm, (255,195,60) if lname else (90,110,170))

    # speed slider
    SBX, SBY, SBW, SBH = slider_rect(H)
    pygame.draw.rect(surf, (35,35,70), (SBX, SBY, SBW, SBH), border_radius=3)
    t = (math.log(speed)-math.log(SPEED_MIN)) / (math.log(SPEED_MAX)-math.log(SPEED_MIN))
    t = max(0.0, min(1.0, t))
    pygame.draw.rect(surf, (70,95,190), (SBX, SBY, int(SBW*t), SBH), border_radius=3)
    hx  = SBX + int(SBW*t)
    hcol = (200,220,255) if dragging_slider else (120,145,220)
    pygame.draw.circle(surf, hcol, (hx, SBY+SBH//2), 8)
    _txt(surf, "SPEED", (SBX, SBY-17), Fsm, (80,80,130))
    _txt(surf, f"{speed:.2f}×", (SBX+SBW+10, SBY-5), Fsm, (130,130,200))

    hints = [
        "RIGHT-DRAG orbit   MID-DRAG pan   SCROLL zoom",
        "LEFT-CLICK planet=lock (again=release)   R reset",
        "SPACE play/pause   ← → step   Q quit",
    ]
    for i, h in enumerate(reversed(hints)):
        _txt(surf, h, (14, H-78-i*17), Fsm, (55,55,95))

    for k in range(K):
        c = tuple(int(x*255) for x in colors[k])
        pygame.draw.circle(surf, c, (W-130, 14+k*19), 5)
        _txt(surf, names[k], (W-120, 7+k*19), Fsm, c)


# ════════════════════════════════════════════════════════════
#  Main loop
# ════════════════════════════════════════════════════════════
def main():
    W, H = 1280, 800
    pygame.init()
    screen = pygame.display.set_mode((W,H), DOUBLEBUF|OPENGL|RESIZABLE)
    pygame.display.set_caption("N-Body Simulator")
    hud = pygame.Surface((W,H), pygame.SRCALPHA)

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
        near = max(span*0.0001, 1e-6)
        gluPerspective(50, w/max(h,1), near, span*300)
        glMatrixMode(GL_MODELVIEW)
    set_proj(W, H)

    frame           = 0
    playing         = True
    speed           = 1.0
    sacc            = 0.0
    locked          = None
    drag            = None
    last_m          = (0, 0)
    dragging_slider = False
    clock           = pygame.time.Clock()

    while True:
        clock.tick(60)

        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit(); sys.exit()

            elif ev.type == VIDEORESIZE:
                W, H = ev.w, ev.h
                screen = pygame.display.set_mode((W,H), DOUBLEBUF|OPENGL|RESIZABLE)
                hud    = pygame.Surface((W,H), pygame.SRCALPHA)
                glViewport(0,0,W,H); set_proj(W,H)

            elif ev.type == KEYDOWN:
                if ev.key in (K_q, K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif ev.key == K_SPACE:
                    playing = not playing
                elif ev.key == K_RIGHT:
                    playing = False; frame = (frame+1) % N
                elif ev.key == K_LEFT:
                    playing = False; frame = (frame-1) % N
                elif ev.key == K_r:
                    cam.reset(); locked = None

            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if on_slider(ev.pos[0], ev.pos[1], H):
                        dragging_slider = True
                        speed = speed_from_mouse(ev.pos[0], H)
                    else:
                        k = pick(ev.pos[0], ev.pos[1], H, frame)
                        if k is not None:
                            locked = None if locked == k else k
                            if locked is not None:
                                cam.target = pos[frame, k].copy()
                        else:
                            locked = None
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
                    dx = ev.pos[0]-last_m[0]; dy = ev.pos[1]-last_m[1]
                    if drag == 'orbit': cam.orbit(dx, dy)
                    else:               cam.pan(dx, dy)
                    last_m = ev.pos

        # ── advance ──────────────────────────────────────────
        if playing:
            sacc  += speed
            steps  = int(sacc); sacc -= steps
            frame  = (frame + steps) % N

        # ── keep camera glued to locked planet ───────────────
        if locked is not None:
            cam.target = pos[frame, locked].copy()

        # ── render ───────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        cam.look()

        draw_stars()

        for k in range(K):
            p   = pos[frame, k]
            col = colors[k]

            # trail
            t0 = max(0, frame-TRAIL)
            draw_trail(pos[t0:frame+1, k], col)

            # get screen-space radius
            use_sphere, draw_r = get_screen_radius(p[0], p[1], p[2], vis_r[k])
            if draw_r is None:
                continue

            if use_sphere:
                # physically accurate size — full sphere + subtle glow
                draw_sphere(p[0],p[1],p[2], draw_r, col, 1.0)
                glDepthMask(GL_FALSE)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                draw_sphere(p[0],p[1],p[2], draw_r*2.6, col, 0.10)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDepthMask(GL_TRUE)
            else:
                # too small — inflated marker, slightly transparent + stronger glow
                draw_sphere(p[0],p[1],p[2], draw_r, col, 0.75)
                glDepthMask(GL_FALSE)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                draw_sphere(p[0],p[1],p[2], draw_r*4.0, col, 0.18)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDepthMask(GL_TRUE)

            # selection ring scaled to draw_r
            if locked == k:
                glPushMatrix()
                glTranslatef(p[0],p[1],p[2])
                r = draw_r*1.8; segs = 56
                glLineWidth(1.8)
                glColor4f(1.0, 0.85, 0.2, 0.9)
                glBegin(GL_LINE_LOOP)
                for s in range(segs):
                    a = 2*math.pi*s/segs
                    glVertex3f(math.cos(a)*r, 0, math.sin(a)*r)
                glEnd()
                glLineWidth(1.0)
                glPopMatrix()

        # ── HUD ──────────────────────────────────────────────
        draw_hud(hud, W, H, frame, playing, locked, speed, dragging_slider)
        raw = pygame.image.tostring(hud, "RGBA", True)
        glWindowPos2i(0, 0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, raw)

        pygame.display.flip()

if __name__ == "__main__":
    main()
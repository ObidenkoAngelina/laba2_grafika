#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math
import random
import argparse
import textwrap
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib

if "--no-gui" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# отключаем стандартные «горячие клавиши» matplotlib, которые мешают управлению
for _k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.fullscreen",
           "keymap.xscale", "keymap.yscale", "keymap.grid", "keymap.grid_minor",
           "keymap.pan", "keymap.zoom", "keymap.save"):
    if _k in plt.rcParams:
        plt.rcParams[_k] = []

EPS = 1e-9
MONO = "DejaVu Sans Mono"


# =====================================================================
#  Геометрия: многоугольник и генерация сцены
# =====================================================================
class Poly:
    """Плоский многоугольник в 3D. Уравнение плоскости ax+by+cz+d=0 - метод Ньюэла."""
    
    # считаем уравнение плоскости по методу Ньюэла чтоб найти глубину z
    def __init__(self, pid, verts, color):
        self.pid = pid                # номер 1 2 3..
        self.verts = verts            # [(x, y, z), ...] вершины
        self.color = color            # (r, g, b) цвет
        a = b = c = 0.0
        n = len(verts)
        for i in range(n):
            x1, y1, z1 = verts[i]
            x2, y2, z2 = verts[(i + 1) % n]
            a += (y1 - y2) * (z1 + z2)
            b += (z1 - z2) * (x1 + x2)
            c += (x1 - x2) * (y1 + y2)
        cx = sum(v[0] for v in verts) / n # находим центр многоугольника.
        cy = sum(v[1] for v in verts) / n
        cz = sum(v[2] for v in verts) / n
        self.a, self.b, self.c = a, b, c
        self.d = -(a * cx + b * cy + c * cz)

    def depth(self, x, y):
        """z = -(ax + by + d)/c"""
        return -(self.a * x + self.b * y + self.d) / self.c

    @property
    def xy(self):
        return [(v[0], v[1]) for v in self.verts]


def _cross(p, q):
    """Вычисляет векторное произведение двух 3D-векторов p и q."""
    return (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])


def make_scene(rng, W, H, n_poly):
    """Случайные плоские многоугольники (3..6 рёбер), протыкающие друг друга."""
    cmap = plt.get_cmap("tab10")
    polys = []
    for pid in range(n_poly):
        k = rng.randint(3, 6)
        R = rng.uniform(0.17, 0.32) * min(W, H)
        # нормаль плоскости; nz >= ~0.5, чтобы плоскость не была параллельна лучу зрения
        nx, ny = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
        ln = math.sqrt(nx * nx + ny * ny + 1.0)
        n = (nx / ln, ny / ln, 1.0 / ln)
        u = (-n[2], 0.0, n[0])
        lu = math.hypot(u[0], u[2])
        u = (u[0] / lu, 0.0, u[2] / lu)
        v = _cross(n, u)
        # углы вершин: многоугольник звёздчатый относительно центра => простой
        while True:
            angs = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            gaps = [angs[i + 1] - angs[i] for i in range(k - 1)] + [angs[0] + 2 * math.pi - angs[-1]]
            if min(gaps) > 0.45 and max(gaps) < 2.6:
                break
        jitter = k >= 4 and rng.random() < 0.5          # иногда делаем невыпуклым
        cx = rng.uniform(R + 1, max(R + 2, W - R - 1))
        cy = rng.uniform(R + 1, max(R + 2, H - R - 1))
        cz = rng.uniform(-25, 25)
        verts = []
        for t in angs:
            rad = R * rng.uniform(0.55, 1.0) if jitter else R
            ct, st = math.cos(t) * rad, math.sin(t) * rad
            verts.append((cx + ct * u[0] + st * v[0],
                          cy + ct * u[1] + st * v[1],
                          cz + ct * u[2] + st * v[2]))
        polys.append(Poly(pid, verts, tuple(cmap(pid)[:3])))
    return polys


# =====================================================================
#  Подготовка данных: таблица рёбер и список активных рёбер (САР)
# =====================================================================
def build_edge_table(polys, H):
    """Группировка рёбер по самой верхней сканирующей строке, которую они пересекают.
    Строка j имеет центр y = j + 0.5, критерий пересечения: ymin <= yc < ymax.
    Горизонтальные рёбра игнорируются."""
    et = defaultdict(list)
    for p in polys:
        pts = p.xy
        m = len(pts)
        for i in range(m):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % m]
            if y1 == y2:
                continue
            if y1 < y2:
                xt, yt, xb, yb = x2, y2, x1, y1
            else:
                xt, yt, xb, yb = x1, y1, x2, y2
            jtop = math.ceil(yt - 0.5) - 1
            jbot = math.ceil(yb - 0.5)
            j0 = min(jtop, H - 1)
            j1 = max(jbot, 0)
            if j0 < j1:
                continue
            slope = (xt - xb) / (yt - yb)                # dx/dy
            x0 = xt + (j0 + 0.5 - yt) * slope
            et[j0].append(dict(pid=p.pid, x=x0, dx=-slope, n=j0 - j1 + 1, new=True))
    return et


def simulate_ael(polys, W, H):
    """Для каждой строки запоминает САР (отсортированный по x) перед её обработкой."""
    et = build_edge_table(polys, H)
    ael, result = [], {}
    for j in range(H - 1, -1, -1):
        for e in ael:
            e["new"] = False
        ael += [dict(e) for e in et.get(j, [])]
        ael.sort(key=lambda e: (e["x"], e["pid"]))
        result[j] = [dict(e) for e in ael]
        nxt = []
        for e in ael: # корректировка САР
            e["n"] -= 1
            if e["n"] > 0:
                e["x"] += e["dx"]
                nxt.append(e)
        ael = nxt
    return result

# (P1, P2...) нужны точные координаты начала и конца каждого отрезка.
def spans_of(polys, j, ael):
    """Отрезки многоугольников в секущей плоскости: (pid, xa, za, xb, zb)."""
    P = {p.pid: p for p in polys}
    yc = j + 0.5
    by = defaultdict(list)
    for e in ael:
        by[e["pid"]].append(e["x"])
    out = []
    for pid, xs in by.items():
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            out.append((pid, xs[i], P[pid].depth(xs[i], yc), xs[i + 1], P[pid].depth(xs[i + 1], yc)))
    return out

# пересекаются или нет фигуры. возвращаем знак.
def sgn(v):
    return 0 if abs(v) < EPS else (1 if v > 0 else -1)


# =====================================================================
#  Сам алгоритм: обработка одной сканирующей строки (с протоколированием шагов)
# =====================================================================
class Scanline:
    """Интервальный алгоритм для одной строки. Если record=True, то каждый значимый шаг сохраняется в self.steps (для демонстрации)."""

    def __init__(self, polys, j, ael, W, record=True):
        self.P = {p.pid: p for p in polys} # словарь всех многоугольников
        self.order = sorted(self.P)
        self.j, self.yc, self.W = j, j + 0.5, W
        self.ael = [dict(e) for e in ael]
        self.rec = record
        self.flags = {q: 0 for q in self.order} # флаги активности.
        self.count = 0
        self.row = np.full(W, -1, dtype=int)         # буфер кадра для одной строки (-1 = фон)
        self.segs = []                               # изображённые отрезки (a, b, pid)
        self.stack = [] # пустой стек (для протыканий).
        self.steps = []
        self.xl, self.xr = 0.0, None # левая и правая границы текущего интервала.
        self.n_splits = 0

    # ---------- служебное ----------
    def z(self, q, x): # для сравнения, кто ближе.
        return self.P[q].depth(x, self.yc)

    #  Закрасить пиксели
    def fill(self, a, b, pid):
        i0 = max(0, math.ceil(a - 0.5))
        i1 = min(self.W, math.ceil(b - 0.5))
        if i1 > i0:
            self.row[i0:i1] = pid
        self.segs.append((a, b, pid))

    # Сделать «фотографию» шага
    def snap(self, title, text, cur=None, interval=None, cand=None, split=None, sub=None, mid=None, ael=None):
        if not self.rec:
            return
        self.steps.append(dict(
            title=title, text=text, cur=cur, interval=interval, cand=cand, split=split,
            sub=sub, mid=mid, ael=ael if ael is not None else list(self.ael),
            flags=dict(self.flags), count=self.count, stack=list(self.stack),
            segs=list(self.segs), row=self.row.copy(), xl=self.xl, xr=self.xr))

    @staticmethod
    def names(ids):
        return ", ".join(f"P{q + 1}" for q in ids) or "-"

    # ---------- основной цикл по строке ----------
    def run(self):
        W = self.W
        if self.ael:
            self.snap("Начало строки",
                      f"Строка y = {self.j}, секущая плоскость y = {self.yc}. Буфер строки ← фон, "
                      f"флаги активности ← 0, счётчик многоугольников ← 0, Xлев ← 0. "
                      f"Список активных рёбер (САР) упорядочен по возрастанию x, в нём {len(self.ael)} "
                      f"рёбер (звёздочка * - ребро только что добавлено из таблицы рёбер). "
                      f"Рёбра делят строку на {len(self.ael) + 1} интервалов.")
        else:
            self.snap("Начало строки",
                      f"Строка y = {self.j}: САР пуст - ни один многоугольник не пересекает строку, "
                      f"вся строка изображается фоном.")

        for idx, e in enumerate(self.ael):
            pid = e["pid"]
            a = self.xl
            self.xr = b = min(max(e["x"], 0.0), float(W))
            act = [q for q in self.order if self.flags[q]]
            head = (f"Извлечено ребро e{idx} (P{pid + 1}): Xправ = {e['x']:.2f}. "
                    f"Интервал [{a:.2f}; {b:.2f}], счётчик многоугольников = {self.count}. ")
            if b - a <= EPS:
                self.snap("Интервал нулевой ширины", head + "Ширина интервала равна 0 - изображать нечего.",
                          cur=idx, interval=(a, b))
            elif self.count == 0: #пустой интервал. Красим фоном.
                self.fill(a, b, -1)
                self.snap("Пустой интервал → фон", head + "Активных многоугольников нет → отрезок изображается фоном.",
                          cur=idx, interval=(a, b))
            elif self.count == 1: # внутри одна фигура. Красим её цветом (глубину считать не надо).
                q = act[0]
                self.fill(a, b, q)
                self.snap("Один отрезок в интервале",
                          head + f"Активен единственный многоугольник P{q + 1} → отрезок изображается его "
                                 f"цветом, глубину считать не нужно.", cur=idx, interval=(a, b))
            else:
                self.snap("Несколько отрезков в интервале",
                          head + f"Активны {self.names(act)} (их {len(act)}) → нужно выбрать видимый. "
                                 f"Сначала проверяем отрезки на пересечение внутри интервала: "
                                 f"sign(z1л − z2л) ≠ sign(z1п − z2п).", cur=idx, interval=(a, b))
                self.resolve(a, b, act, idx)

            # смена признака активности многоугольника (левое/правое ребро)
            old = self.flags[pid]
            self.flags[pid] = 1 - old
            # Если мы встретили левое ребро — счетчик +1 (мы вошли в фигуру). Если правое — счетчик -1 (вышли).
            self.count += 1 if old == 0 else -1
            self.xl = self.xr
            role = "левое (многоугольник становится активным)" if old == 0 else \
                "правое (многоугольник перестаёт быть активным)"
            self.snap("Смена флага активности",
                      f"Ребро e{idx} многоугольника P{pid + 1} - {role}: флаг {old} → {1 - old}, "
                      f"счётчик многоугольников = {self.count}. Xлев ← Xправ = {self.xl:.2f}.", cur=idx)

        if self.xl < W - EPS:
            a = self.xl
            self.xr = float(W)
            self.fill(a, float(W), -1)
            self.snap("Остаток строки → фон",
                      f"Активных рёбер больше нет, Xлев = {a:.2f} < Xmax = {W}. Последний интервал "
                      f"[{a:.2f}; {W}] изображается фоном.", interval=(a, float(W)))
        # корректировка САР
        upd = []
        for e in self.ael:
            ne = dict(e)
            ne["x_new"] = e["x"] + e["dx"]
            ne["n_new"] = e["n"] - 1
            upd.append(ne)
        gone = sum(1 for e in upd if e["n_new"] <= 0)
        self.snap("Корректировка САР",
                  f"Строка обработана и выведена на экран. Корректировка САР для следующей строки "
                  f"y = {self.j - 1}: x ← x + Δx, n ← n − 1; рёбра с n = 0 удаляются "
                  f"(таких рёбер: {gone}). Многоугольник без рёбер покидает список активных.",
                  ael=upd)
        return self

    # ---------- обработка интервала с несколькими отрезками ----------
    def resolve(self, a, b, act, idx):
        outer = (a, b)
        self.stack = []
        for _ in range(200):
            zl = {q: self.z(q, a) for q in act}
            zr = {q: self.z(q, b) for q in act}
            cand = [(q, zl[q], zr[q]) for q in act]
            best, lines = None, []
            for q1, q2 in combinations(act, 2):
                dl, dr = zl[q1] - zl[q2], zr[q1] - zr[q2]
                s1, s2 = sgn(dl), sgn(dr)
                lines.append(f"P{q1 + 1}/P{q2 + 1}: sign(z1л−z2л) = {s1:+d}, sign(z1п−z2п) = {s2:+d}")
                if s1 * s2 < 0:
                    xi = a + (b - a) * dl / (dl - dr)
                    if best is None or xi < best[0]:
                        best = (xi, q1, q2)
            sub = (a, b)
            if best is not None:
                xi, q1, q2 = best
                self.n_splits += 1
                zi = self.z(q1, xi)
                self.stack.append(b)
                self.snap("Пересечение отрезков → разбиение",
                          f"Подынтервал [{a:.2f}; {b:.2f}]. " + "; ".join(lines[:3]) +
                          f". Знаки разные → P{q1 + 1} и P{q2 + 1} пересекаются в x_i = {xi:.2f}, z = {zi:.2f}. "
                          f"Правая граница {b:.2f} заносится в стек, Xправ ← {xi:.2f}, "
                          f"дальше обрабатывается левый подынтервал [{a:.2f}; {xi:.2f}].",
                          cur=idx, interval=outer, cand=cand, split=xi, sub=sub)
                b = xi
                continue
            xm = 0.5 * (a + b)
            zm = {q: self.z(q, xm) for q in act}
            vis = max(act, key=lambda q: zm[q])
            self.fill(a, b, vis)
            tail = ""
            if self.stack:
                tail = (f" Из стека будет извлечена правая граница {self.stack[-1]:.2f}: "
                        f"следующий подынтервал [{b:.2f}; {self.stack[-1]:.2f}].")
            self.snap("Подынтервал без пересечений: z в центре",
                      f"В [{a:.2f}; {b:.2f}] отрезки не пересекаются. Глубина считается в середине "
                      f"x = {xm:.2f}: " + ", ".join(f"z(P{q + 1}) = {zm[q]:.2f}" for q in act) +
                      f" → видим P{vis + 1} (максимальное z - ближе к наблюдателю). Отрезок изображён." + tail,
                      cur=idx, interval=outer, cand=cand, sub=sub, mid=(xm, zm))
            if not self.stack:
                break
            a, b = b, self.stack.pop()


# =====================================================================
#  Сцена целиком: прогон алгоритма по всем строкам
# =====================================================================
class Scene:
    def __init__(self, seed, W, H, n_poly=None):
        self.seed, self.W, self.H = seed, W, H
        rng = random.Random(seed)
        for _ in range(300): # хотя бы одно протыкание
            n = n_poly or rng.choice((5, 6))
            self.polys = make_scene(rng, W, H, n)
            self._compute()
            if sum(self.splits.values()) > 0:
                break

    def _compute(self):
        W, H = self.W, self.H
        self.ael_at = simulate_ael(self.polys, W, H)
        self.runs, self.splits = {}, {}
        self.final = np.full((H, W), -1, dtype=int)
        for j in range(H):
            r = Scanline(self.polys, j, self.ael_at[j], W, record=True).run()
            self.runs[j] = r
            self.splits[j] = r.n_splits
            self.final[j] = r.row
        self.spans = {j: spans_of(self.polys, j, self.ael_at[j]) for j in range(H)}

    # найти где пересекаются
    def interesting_line(self):
        j = max(self.splits, key=lambda k: (self.splits[k], k))
        if self.splits[j] > 0:
            return j
        for j in range(self.H - 1, -1, -1):
            if self.ael_at[j]:
                return j
        return self.H - 1


# =====================================================================
#  Визуализация
# =====================================================================
HELP = ("→/Пробел шаг  ←  назад | ↓/↑ строка (или клик по картинке) | f конец строки | a авто (+/- скорость) | "
        "i строка с протыканием | t итог | w каркас | n новая сцена | q выход")


class Demo:
    def __init__(self, scene, args):
        self.args = args
        self.delay = 500
        self.timer = None
        self.wire = False
        self.show_final = False
        self.fig = plt.figure(figsize=(18, 9.6))
        gs = self.fig.add_gridspec(4, 3, width_ratios=[1.05, 1.4, 0.95], height_ratios=[1, 1, 1, 0.45])
        self.ax_img = self.fig.add_subplot(gs[0:3, 0])
        self.ax_row = self.fig.add_subplot(gs[3, 0])
        self.ax_plane = self.fig.add_subplot(gs[0:2, 1])
        self.ax_log = self.fig.add_subplot(gs[2:4, 1])
        self.ax_ael = self.fig.add_subplot(gs[:, 2])
        self.fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.13, wspace=0.14, hspace=0.45)
        self.fig.suptitle("Интервальный алгоритм построчного сканирования - вариант 5",
                          fontsize=13, fontweight="bold")
        self.fig.text(0.01, 0.012, HELP, fontsize=9, color="0.25")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.load_scene(scene)

    # ---------- сцена / навигация ----------
    def load_scene(self, scene):
        self.sc = scene
        self.W, self.H = scene.W, scene.H
        cols = [(0.72, 0.72, 0.72), (1, 1, 1)] + [p.color for p in scene.polys]
        self.pal = np.array(cols)
        self.j = scene.interesting_line()
        self.k = 0
        self.redraw()

    def steps(self):
        return self.sc.runs[self.j].steps

    def goto_line(self, j, direction=-1, last=False):
        H = self.H
        j = min(max(j, 0), H - 1)
        while 0 <= j < H and not self.sc.ael_at[j]:
            j += direction
        if not (0 <= j < H):
            return False
        self.j = j
        self.k = len(self.steps()) - 1 if last else 0
        self.redraw()
        return True

    def next_step(self):
        if self.k < len(self.steps()) - 1:
            self.k += 1
            self.redraw()
            return True
        return self.goto_line(self.j - 1, -1)

    def prev_step(self):
        if self.k > 0:
            self.k -= 1
            self.redraw()
        else:
            self.goto_line(self.j + 1, +1, last=True)

    def tick(self):
        if not self.next_step():
            self.toggle_auto()

    def toggle_auto(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        else:
            self.timer = self.fig.canvas.new_timer(interval=self.delay)
            self.timer.add_callback(self.tick)
            self.timer.start()

    def set_delay(self, factor):
        self.delay = int(min(3000, max(30, self.delay * factor)))
        if self.timer is not None:
            self.timer.interval = self.delay

    def on_key(self, ev):
        k = ev.key
        if k in ("right", " "):
            self.next_step()
        elif k == "left":
            self.prev_step()
        elif k == "down":
            self.goto_line(self.j - 1, -1)
        elif k == "up":
            self.goto_line(self.j + 1, +1)
        elif k == "f":
            self.k = len(self.steps()) - 1
            self.redraw()
        elif k == "a":
            self.toggle_auto()
        elif k in ("+", "="):
            self.set_delay(0.6)
        elif k in ("-", "_"):
            self.set_delay(1.6)
        elif k == "i":
            self.j, self.k = self.sc.interesting_line(), 0
            self.redraw()
        elif k == "t":
            self.show_final = not self.show_final
            self.redraw()
        elif k == "w":
            self.wire = not self.wire
            self.redraw()
        elif k == "n":
            self.load_scene(Scene(self.sc.seed + 1, self.W, self.H, self.args.polygons))

    def on_click(self, ev):
        if ev.inaxes is self.ax_img and ev.ydata is not None:
            self.goto_line(int(ev.ydata), -1)

    # ---------- отрисовка ----------
    def col(self, pid):
        return self.sc.polys[pid].color

    def redraw(self):
        run = self.sc.runs[self.j]
        st = run.steps[self.k]
        self.draw_image(st)
        self.draw_row(st)
        self.draw_plane(st)
        self.draw_log(run, st)
        self.draw_ael(st)
        self.fig.canvas.draw_idle()

    def draw_image(self, st):
        ax, W, H, j = self.ax_img, self.W, self.H, self.j
        ax.clear()
        if self.show_final:
            proc = self.sc.final.copy()
        else:
            proc = np.full((H, W), -2, dtype=int)
            proc[j + 1:] = self.sc.final[j + 1:]
            proc[j] = st["row"]
        ax.imshow(self.pal[proc + 2], origin="lower", extent=(0, W, 0, H),
                  interpolation="nearest", aspect="equal")
        yc = j + 0.5
        if not self.show_final:
            ax.axhline(yc, color="red", lw=1.3)
            if st["interval"] is not None:
                ax.plot(st["interval"], [yc, yc], color="orange", lw=5, solid_capstyle="butt")
            if st["sub"] is not None:
                ax.plot(st["sub"], [yc, yc], color="deepskyblue", lw=2.5, solid_capstyle="butt")
        if self.wire:
            for p in self.sc.polys:
                pts = p.xy + [p.xy[0]]
                ax.plot([t[0] for t in pts], [t[1] for t in pts], color="k", lw=0.8, alpha=0.6)
        for p in self.sc.polys:
            xs, ys = zip(*p.xy)
            ax.text(sum(xs) / len(xs), sum(ys) / len(ys), f"P{p.pid + 1}", fontsize=8,
                    ha="center", va="center", color="k", alpha=0.7)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_title("Итоговое изображение" if self.show_final else
                     f"Изображение (строки сверху вниз) - строка y = {j}", fontsize=10)

    def draw_row(self, st):
        ax = self.ax_row
        ax.clear()
        W = self.W
        ax.imshow(self.pal[st["row"][None, :] + 2], aspect="auto", interpolation="nearest",
                  extent=(0, W, 0, 1))
        for x in range(0, W, 8):
            ax.axvline(x, color="k", lw=0.3, alpha=0.4)
        ax.set_yticks([])
        ax.set_xlim(0, W)
        ax.set_title("Буфер кадра для строки (белый - фон)", fontsize=9)

    def draw_plane(self, st):
        ax, W = self.ax_plane, self.W
        ax.clear()
        spans = self.sc.spans[self.j]
        zs = [s[2] for s in spans] + [s[4] for s in spans]
        zmin, zmax = (min(zs), max(zs)) if zs else (0.0, 1.0)
        rng_ = max(zmax - zmin, 1.0)
        top = zmax + 0.32 * rng_
        bot = zmin - 0.28 * rng_
        # границы интервалов и их номера
        xs_e = [e["x"] for e in st["ael"]]
        bounds = [0.0] + sorted(min(max(x, 0.0), W) for x in xs_e) + [float(W)]
        for i in range(len(bounds) - 1):
            ax.text(0.5 * (bounds[i] + bounds[i + 1]), top - 0.17 * rng_, f"{i + 1}", ha="center",
                    va="top", fontsize=8, color="0.45", style="italic")
        # подсветка
        if st["interval"] is not None:
            ax.axvspan(st["interval"][0], st["interval"][1], color="orange", alpha=0.22, lw=0)
        if st["sub"] is not None:
            ax.axvspan(st["sub"][0], st["sub"][1], color="deepskyblue", alpha=0.25, lw=0)
        for idx, e in enumerate(st["ael"]):
            cur = (idx == st["cur"])
            ax.axvline(e["x"], color="red" if cur else "0.6", ls="-" if cur else "--",
                       lw=1.8 if cur else 0.9)
            ax.text(e["x"], top - 0.04 * rng_, f"e{idx}", ha="center", va="top", fontsize=8,
                    color="red" if cur else "0.35", fontweight="bold" if cur else "normal")
        # отрезки многоугольников в секущей плоскости
        for pid, xa, za, xb, zb in spans:
            ax.plot([xa, xb], [za, zb], color=self.col(pid), lw=3.2, solid_capstyle="butt")
            ax.text(0.5 * (xa + xb), 0.5 * (za + zb) + 0.04 * rng_, f"P{pid + 1}",
                    color=self.col(pid), fontsize=9, ha="center", fontweight="bold")
        # глубины на концах подынтервала / в центре
        if st["cand"] is not None and st["sub"] is not None:
            a, b = st["sub"]
            for q, zl, zr in st["cand"]:
                ax.plot([a, b], [zl, zr], "o", color=self.col(q), mec="k", ms=6, zorder=5)
                ax.annotate(f"{zl:.1f}", (a, zl), textcoords="offset points", xytext=(-4, 6),
                            fontsize=7, ha="right")
                ax.annotate(f"{zr:.1f}", (b, zr), textcoords="offset points", xytext=(4, 6), fontsize=7)
        if st["mid"] is not None:
            xm, zm = st["mid"]
            ax.axvline(xm, color="blue", ls=":", lw=1)
            for q, z in zm.items():
                ax.plot([xm], [z], "s", color=self.col(q), mec="k", ms=7, zorder=6)
        if st["split"] is not None:
            ax.axvline(st["split"], color="magenta", ls="-.", lw=1.6)
            ax.text(st["split"], bot + 0.03 * rng_, f"x_i={st['split']:.2f}", color="magenta",
                    fontsize=8, ha="center", va="bottom", fontweight="bold")
        ax.set_xlim(0, W)
        ax.set_ylim(bot, top)
        ax.set_xlabel("x (пиксели)")
        ax.set_ylabel("z  (выше - ближе к наблюдателю)")
        ax.grid(alpha=0.2)
        ax.set_title(f"Секущая плоскость y = {self.j + 0.5}   (оранжевый - интервал, голубой - подинтервал, "
                     f"фиолетовый - точка пересечения)", fontsize=9)

    def draw_log(self, run, st):
        ax = self.ax_log
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        n = len(run.steps)
        ax.text(0, 0.85, f"Шаг {self.k + 1}/{n} строки y={self.j}: {st['title']}", fontsize=11.5,
                fontweight="bold", va="top")
        ax.text(0, 0.75, textwrap.fill(st["text"], 88), fontsize=10, va="top")

    def draw_ael(self, st):
        ax = self.ax_ael
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        state = {"y": 0.99}
        dy = 0.0255

        def put(text, color="black", weight="normal", bg=None, gap=0.0):
            state["y"] -= gap
            ax.text(0.0, state["y"], text, family=MONO, fontsize=8.6, va="top", color=color,
                    fontweight=weight,
                    bbox=dict(facecolor=bg, edgecolor="none", pad=1.0) if bg else None)
            state["y"] -= dy

        upd = any("x_new" in e for e in st["ael"])
        put("СПИСОК АКТИВНЫХ РЁБЕР (по возр. x)", weight="bold")
        if upd:
            put("  №  полиг.     x → x_нов     Δx    n→n-1")
        else:
            put("  №  полиг.       x         Δx     n")
        if not st["ael"]:
            put("  (пусто)", color="0.4")
        for idx, e in enumerate(st["ael"][:16]):
            mark = "*" if e.get("new") else " "
            if upd:
                rem = "  ✗" if e["n_new"] <= 0 else ""
                line = (f" e{idx:<2d} P{e['pid'] + 1}{mark}  {e['x']:7.2f}→{e['x_new']:7.2f} "
                        f"{e['dx']:6.2f}  {e['n']}→{e['n_new']}{rem}")
            else:
                line = f" e{idx:<2d} P{e['pid'] + 1}{mark}     {e['x']:7.2f}   {e['dx']:7.2f}   {e['n']:3d}"
            cur = (idx == st["cur"])
            put(line, color=self.col(e["pid"]) if not cur else "black",
                weight="bold" if cur else "normal", bg="#fff0a0" if cur else None)
        if len(st["ael"]) > 16:
            put(f" ... ещё {len(st['ael']) - 16}", color="0.4")

        put("ФЛАГИ АКТИВНОСТИ МНОГОУГОЛЬНИКОВ", weight="bold", gap=0.01)
        for q, v in st["flags"].items():
            put(f"  P{q + 1}: {v}   {'■ активен' if v else '□'}", color=self.col(q) if v else "0.55",
                weight="bold" if v else "normal")
        put(f"Счётчик многоугольников = {st['count']}", weight="bold", gap=0.01)
        xr = "-" if st["xr"] is None else f"{st['xr']:.2f}"
        put(f"Xлев = {st['xl']:.2f}     Xправ = {xr}")
        put("Стек пересечений: " + ("[" + ", ".join(f"{v:.2f}" for v in st["stack"]) + "]"
                                      if st["stack"] else "пуст"))

    def save_frame(self, path):
        self.fig.savefig(path, dpi=80)


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Интервальный алгоритм построчного сканирования (вариант 5)")
    ap.add_argument("--seed", type=int, default=None, help="зерно генератора сцены")
    ap.add_argument("--polygons", type=int, choices=(5, 6), default=None, help="число многоугольников")
    ap.add_argument("--width", type=int, default=80, help="ширина растра (пикселов)")
    ap.add_argument("--height", type=int, default=60, help="высота растра (пикселов)")
    ap.add_argument("--no-gui", action="store_true", help="без окна (только сохранить PNG, см. --save)")
    ap.add_argument("--save", metavar="PREFIX", help="сохранить PNG-кадры с префиксом PREFIX")
    args = ap.parse_args()

    # Отключаем панель инструментов внизу окна (лупа, крестик, координаты)
    import matplotlib
    matplotlib.rcParams['toolbar'] = 'None'

    # Создаем сцену
    seed = args.seed if args.seed is not None else random.randrange(10 ** 6)
    sc = Scene(seed, args.width, args.height, args.polygons)

    # Запускаем визуализацию
    demo = Demo(sc, args)
    
    if args.save:
        demo.save_frame(f"{args.save}_step_first.png")
        demo.k = len(demo.steps()) // 2
        demo.redraw()
        demo.save_frame(f"{args.save}_step_mid.png")
        demo.k = len(demo.steps()) - 1
        demo.redraw()
        demo.save_frame(f"{args.save}_step_last.png")
        demo.show_final = True
        demo.redraw()
        demo.save_frame(f"{args.save}_final.png")

    if not args.no_gui:
        plt.show()

if __name__ == "__main__":
    main()
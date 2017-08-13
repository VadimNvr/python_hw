import Quartz.CoreGraphics as CG
import struct
import matplotlib.pyplot as plt
from AppKit import NSEvent
from itertools import product
from datetime import time
import time
import json
import math
import datetime
from math import pi
import numpy as np
from Quartz import CGDisplayPixelsHigh
from Quartz.CoreGraphics import CGEventCreate
from Quartz.CoreGraphics import CGEventGetLocation
from Quartz.CoreGraphics import CGEventCreateMouseEvent
from Quartz.CoreGraphics import kCGEventMouseMoved
from Quartz.CoreGraphics import CGEventCreateKeyboardEvent
from Quartz.CoreGraphics import CGEventPost
from Quartz.CoreGraphics import kCGEventMouseMoved
from Quartz.CoreGraphics import kCGEventLeftMouseDown
from Quartz.CoreGraphics import kCGEventLeftMouseUp
from Quartz.CoreGraphics import kCGEventRightMouseDown
from Quartz.CoreGraphics import kCGEventRightMouseUp
from Quartz.CoreGraphics import kCGMouseButtonLeft
from Quartz.CoreGraphics import kCGMouseButtonRight
from Quartz.CoreGraphics import kCGHIDEventTap
from Quartz.CoreGraphics import CGDisplayShowCursor
from Quartz.CoreGraphics import CGAssociateMouseAndMouseCursorPosition


def sign(val):
    if val < 0:
        return -1
    elif val > 0:
        return 1
    else:
        return 0


keyCodeMap = {
    'a'                 : 0x00,
    's'                 : 0x01,
    'd'                 : 0x02,
    'f'                 : 0x03,
    'h'                 : 0x04,
    'g'                 : 0x05,
    'z'                 : 0x06,
    'x'                 : 0x07,
    'c'                 : 0x08,
    'v'                 : 0x09,
    'b'                 : 0x0B,
    'q'                 : 0x0C,
    'w'                 : 0x0D,
    'e'                 : 0x0E,
    'r'                 : 0x0F,
    'y'                 : 0x10,
    't'                 : 0x11,
    '1'                 : 0x12,
    '2'                 : 0x13,
    '3'                 : 0x14,
    '4'                 : 0x15,
    '6'                 : 0x16,
    '5'                 : 0x17,
    '='                 : 0x18,
    '9'                 : 0x19,
    '7'                 : 0x1A,
    '-'                 : 0x1B,
    '8'                 : 0x1C,
    '0'                 : 0x1D,
    ']'                 : 0x1E,
    'o'                 : 0x1F,
    'u'                 : 0x20,
    '['                 : 0x21,
    'i'                 : 0x22,
    'p'                 : 0x23,
    'l'                 : 0x25,
    'j'                 : 0x26,
    '\''                : 0x27,
    'k'                 : 0x28,
    ';'                 : 0x29,
    '\\'                : 0x2A,
    ','                 : 0x2B,
    '/'                 : 0x2C,
    'n'                 : 0x2D,
    'm'                 : 0x2E,
    '.'                 : 0x2F,
    '`'                 : 0x32,
    'k.'                : 0x41,
    'k*'                : 0x43,
    'k+'                : 0x45,
    'kclear'            : 0x47,
    'k/'                : 0x4B,
    'k\n'               : 0x4C,
    'k-'                : 0x4E,
    'k='                : 0x51,
    'k0'                : 0x52,
    'k1'                : 0x53,
    'k2'                : 0x54,
    'k3'                : 0x55,
    'k4'                : 0x56,
    'k5'                : 0x57,
    'k6'                : 0x58,
    'k7'                : 0x59,
    'k8'                : 0x5B,
    'k9'                : 0x5C,

    # keycodes for keys that are independent of keyboard layout
    '\n'                : 0x24,
    '\t'                : 0x30,
    ' '                 : 0x31,
    'del'               : 0x33,
    'delete'            : 0x33,
    'esc'               : 0x35,
    'escape'            : 0x35,
    'cmd'               : 0x37,
    'command'           : 0x37,
    'shift'             : 0x38,
    'caps lock'         : 0x39,
    'option'            : 0x3A,
    'ctrl'              : 0x3B,
    'control'           : 0x3B,
    'right shift'       : 0x3C,
    'rshift'            : 0x3C,
    'right option'      : 0x3D,
    'roption'           : 0x3D,
    'right control'     : 0x3E,
    'rcontrol'          : 0x3E,
    'fun'               : 0x3F,
    'function'          : 0x3F,
    'f17'               : 0x40,
    'volume up'         : 0x48,
    'volume down'       : 0x49,
    'mute'              : 0x4A,
    'f18'               : 0x4F,
    'f19'               : 0x50,
    'f20'               : 0x5A,
    'f5'                : 0x60,
    'f6'                : 0x61,
    'f7'                : 0x62,
    'f3'                : 0x63,
    'f8'                : 0x64,
    'f9'                : 0x65,
    'f11'               : 0x67,
    'f13'               : 0x69,
    'f16'               : 0x6A,
    'f14'               : 0x6B,
    'f10'               : 0x6D,
    'f12'               : 0x6F,
    'f15'               : 0x71,
    'help'              : 0x72,
    'home'              : 0x73,
    'pgup'              : 0x74,
    'page up'           : 0x74,
    'forward delete'    : 0x75,
    'f4'                : 0x76,
    'end'               : 0x77,
    'f2'                : 0x78,
    'page down'         : 0x79,
    'pgdn'              : 0x79,
    'f1'                : 0x7A,
    'left'              : 0x7B,
    'right'             : 0x7C,
    'down'              : 0x7D,
    'up'                : 0x7E
}


def mouseEvent(type, *args):
    if len(args) == 2:
        posx, posy = args
    elif len(args) == 1 and isinstance(args[0], Point):
        posx, posy = args[0]
    else:
        raise Exception('Error! Invalid arguments for mouseEvent')

    CGEventPost(kCGHIDEventTap, CGEventCreateMouseEvent(
        None,
        type,
        (posx, posy),
        kCGMouseButtonLeft
    ))
    time.sleep(0.01)

def mousemove(*args):
    mouseEvent(kCGEventMouseMoved, *args)


def mouseclick(*args):
    mouseEvent(kCGEventLeftMouseDown, *args);
    mouseEvent(kCGEventLeftMouseUp, *args);


def mousepress(*args):
    mouseEvent(kCGEventRightMouseDown, *args)


def mouserelease(*args):
    mouseEvent(kCGEventRightMouseUp, *args)


def mouseposition():
    loc = NSEvent.mouseLocation()
    return Point(loc.x, CGDisplayPixelsHigh(0) - loc.y)


SCREENSHOT_WIDTHS = [2 ** i for i in range(11)] + [1440]


def screenshot(*args):
    if len(args) == 4:
        x, y, w, h = args
    elif len(args) == 3 and isinstance(args[0], Point):
        x, y = args[0]
        w, h = args[1:]
    else:
        raise Exception('Error! Bad input to screenshot.')

    norm_w = SCREENSHOT_WIDTHS[np.searchsorted(SCREENSHOT_WIDTHS, w)]

    region = CG.CGRectMake(x, y, norm_w, h)

    image = CG.CGWindowListCreateImage(
        region,
        CG.kCGWindowListOptionOnScreenOnly,
        CG.kCGNullWindowID,
        CG.kCGWindowImageDefault
    )

    provider = CG.CGImageGetDataProvider(image)
    bin_data = CG.CGDataProviderCopyData(provider)

    data_format = "%dB" % norm_w * h * 4
    unpacked    = struct.unpack_from(data_format, bin_data)

    if norm_w == w:
        return np.array(unpacked, dtype=np.uint8).reshape(h, w, 4)
    else:
        return np.array(unpacked, dtype=np.uint8).reshape(h, norm_w, 4)[:, :w]


class Dir(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, val):
        if isinstance(val, Point):
            return Point(self.x + val.x, self.y + val.y)
        elif isinstance(val, Dir):
            return Dir(self.x + val.x, self.y + val.y)
        else:
            return Dir(self.x + val, self.y + val)

    def __sub__(self, val):
        if isinstance(val, Point):
            raise Exception('Error! Can not sub point from dir.')
        elif isinstance(val, Dir):
            return Dir(self.x - val.x, self.y - val.y)
        else:
            return Dir(self.x - val, self.y - val)

    def __mul__(self, val):
        return Dir(self.x * val, self.y * val)

    def __floordiv__(self, val):
        return Dir(self.x / val, self.y / val)

    def __truediv__(self, val):
        return Dir(self.x / val, self.y / val)

    def __iter__(self):
        return iter((self.x, self.y))

    def __str__(self):
        return 'Dir({}, {})'.format(self.x, self.y)

    def l1(self):
        return abs(self.x) + abs(self.y)

    def l2(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self):
        if self.x == 0 and self.y == 0:
            return self
        return self / self.l2()


class Point(object):
    def __init__(self, x, y):
        self.x = x if isinstance(x, int) else int(round(x))
        self.y = y if isinstance(y, int) else int(round(y))

    def __add__(self, val):
        if isinstance(val, (Point, Dir)):
            return Point(self.x + val.x, self.y + val.y)
        else:
            return Point(self.x + val, self.y + val)

    def __sub__(self, val):
        if isinstance(val, Point):
            return Dir(self.x - val.x, self.y - val.y)
        elif isinstance(val, Dir):
            return Point(self.x - val.x, self.y - val.y)
        else:
            return Point(self.x - val, self.y - val)

    def __eq__(self, pos):
        return self.x == pos.x and self.y == pos.y

    def __mul__(self, val):
        raise Exception('Error! Can not multiply point.')

    def __div__(self, val):
        raise Exception('Error! Can not divide point.')

    def __iter__(self):
        return iter((self.x, self.y))

    def __str__(self):
        return 'Point({}, {})'.format(self.x, self.y)


class FarmPoint(object):
    def __init__(self, pos, resource):
        self.pos = pos
        self.resource = resource

class WayPoint(object):
    def __init__(self, pos):
        self.pos = pos


class MapArea(object):
    PLAYER_COLOR = (255, 179, 97, 255)

    def __init__(self, data):
        self.data = data

    def find_player(self):
        y_coords, x_coords = np.where((self.data == self.PLAYER_COLOR).all(axis=2))

        if y_coords.size == 0 or x_coords.size == 0:
            plt.figure(figsize=(20, 20))
            plt.imshow(self.data)
            plt.grid()
            plt.show()

            raise Exception('Could not find player')

        return Point(np.mean(x_coords), np.mean(y_coords))


class Map(object):
    BASE_OFFSET = Point(0, 100)
    AROUND_SIZE = 128

    @staticmethod
    def full():
        return MapArea(screenshot(Map.BASE_OFFSET, 1440, 900))

    @staticmethod
    def around(point):
        offset = point - Map.AROUND_SIZE // 2
        return offset, MapArea(screenshot(offset + Map.BASE_OFFSET, Map.AROUND_SIZE, Map.AROUND_SIZE))


def grayscale(img):
    return np.dot(img[:,:,:3], [0.299, 0.587, 0.114])


def normalize(gs):
    return gs / np.max(gs)


def level(gs, lvl=0.5):
    gs[gs <= lvl] = 0
    gs[gs >  lvl] = 1
    return gs


def pad(gs):
    return np.lib.pad(gs, (1,), 'constant', constant_values=(0,))


class Resource(object):
    FARM_OFFSET = Point(590, 524)
    HINT_OFFSET = Point(36, 34)

    def __init__(self):
        self.farm_img = self.get_farm_img()
        self.hint_img = self.get_hint_img()
        self.hint_errors = np.zeros(self.hint_img.shape)
        for y, x in product(range(self.hint_img.shape[0]), range(self.hint_img.shape[1])):
            self.hint_errors[y, x] = Resource.dist(self.hint_img, x, y)
        self.hint_errors = self.hint_errors ** 4

    @staticmethod
    def dist(img, x, y):
        for d in range(min(img.shape)):
            ymin, ymax = max(0, y-d), y+d+1
            xmin, xmax = max(0, x-d), x+d+1

            if 255 in img[ymin:ymax, xmin:xmax]:
                return d

        return min(img.shape)

    def get_farm_img(self):
        with open('/Users/vadim/albion_bot/resources/properties.json', 'r') as f:
            properties = json.load(f)['%s_farm.bin' % self.name()]

        with open('/Users/vadim/albion_bot/resources/%s_farm.bin' % self.name()) as f:
            return np.fromfile(f, dtype=np.uint8).reshape(*properties['shape'])

    def get_hint_img(self):
        with open('/Users/vadim/albion_bot/resources/properties.json', 'r') as f:
            properties = json.load(f)['%s_hint.bin' % self.name()]

        with open('/Users/vadim/albion_bot/resources/%s_hint.bin' % self.name()) as f:
            return np.fromfile(f, dtype=np.uint8).reshape(*properties['shape'])

    def name(self):
        raise Exception("Not implemented")

    @staticmethod
    def process_hint(img):
        return level(normalize(grayscale(img)))


class Iron3(Resource):

    def __init__(self):
        super(Iron3, self).__init__()

    def name(self):
        return 'iron_3'

    def hint_compare(self, img):
        padded = pad(img)

        origin = img
        tl = padded[:-2,:-2]
        tr = padded[:-2,2:]
        bl = padded[2:,:-2]
        br = padded[2:,2:]

        inline_etalon = self.hint_img.reshape(-1)

        res = []
        for tgt in (origin, tl, tr, bl, br):
            inline_tgt = tgt.reshape(-1)

            correct = len(np.where((inline_etalon > 0.5) & (inline_tgt > 0.5))[0]) + \
                      len(np.where((inline_etalon < 0.5) & (inline_tgt < 0.5))[0]) * 0.5
            failure = len(np.where((inline_etalon < 0.5) & (inline_tgt > 0.5))[0])

            res.append(correct - failure)

        return max(res)

    def hint_matches(self, img):
        return self.hint_compare(img) > 160


class Wood2(Resource):

    def __init__(self):
        super(Wood2, self).__init__()

    def name(self):
        return 'wood_2'

    def hint_compare(self, img):
        padded = pad(img)

        origin = img
        tl = padded[:-2,:-2]
        tr = padded[:-2,2:]
        bl = padded[2:,:-2]
        br = padded[2:,2:]
        level = 0.5

        inline_etalon = self.hint_img.reshape(-1)
        inline_errors = self.hint_errors.reshape(-1)
        etalon_positive_cnt = len(np.where(inline_etalon > level)[0])

        res = []
        for tgt in (origin, tl, tr, bl, br):
            inline_tgt = tgt.reshape(-1)

            correct = len(np.where((inline_etalon > level) & (inline_tgt > level))[0]) + \
                      len(np.where((inline_etalon < level) & (inline_tgt < level))[0]) * 0.5

            failure_indices = np.where((inline_etalon < level) & (inline_tgt > level))[0]
            failure = np.sum(inline_errors[failure_indices])

            tgt_positive_cnt = len(np.where(inline_tgt > level)[0])
            failure += max(abs(tgt_positive_cnt - etalon_positive_cnt) - 20, 0) ** 4

            res.append(correct - failure)

        return max(res)

    def hint_matches(self, img):
        return self.hint_compare(img) > 190

class Wood3(Resource):

    def __init__(self):
        super(Wood3, self).__init__()

    def name(self):
        return 'wood_3'

    def hint_compare(self, img):
        padded = pad(img)

        origin = img
        tl = padded[:-2,:-2]
        tr = padded[:-2,2:]
        bl = padded[2:,:-2]
        br = padded[2:,2:]
        level = 0.5

        inline_etalon = self.hint_img.reshape(-1)
        inline_errors = self.hint_errors.reshape(-1)
        etalon_positive_cnt = len(np.where(inline_etalon > level)[0])

        res = []
        for tgt in (origin, tl, tr, bl, br):
            inline_tgt = tgt.reshape(-1)

            correct = len(np.where((inline_etalon > level) & (inline_tgt > level))[0]) + \
                      len(np.where((inline_etalon < level) & (inline_tgt < level))[0]) * 0.5

            failure_indices = np.where((inline_etalon < level) & (inline_tgt > level))[0]
            failure = np.sum(inline_errors[failure_indices])

            tgt_positive_cnt = len(np.where(inline_tgt > level)[0])
            failure += max(abs(tgt_positive_cnt - etalon_positive_cnt) - 20, 0) ** 4

            res.append(correct - failure)

        return max(res)

    def hint_matches(self, img):
        return self.hint_compare(img) > 190


class AlbionAutoMove(object):
    DEFAULT_WAIT = 2

    def __init__(self):
        self.points = []
        self.next_point = None
        self.points_iterator = None

        self.map = None
        self.player_pos = None
        self.center = Point(1440 // 2, 900 * 4 // 9)

        self.map_opened = False
        self.mounted = True

    def wait_action(self, action_img, misses_available=1):
        missed_actions = 0

        while missed_actions <= misses_available:
            current_action_img = screenshot(
                Resource.FARM_OFFSET,
                32, 32
            )

            matched = np.where((action_img == current_action_img).all(axis=2))[0]
            if (matched.size < current_action_img.size // 8):
                global _1
                _1 = (action_img == current_action_img).all(axis=2)
                missed_actions += 1
            else:
                missed_actions = 0
            time.sleep(0.5)

    def get_side_by_dir(self, move_direction):
        x, y = move_direction
        y = -y

        print('Direction: ', move_direction)

        if x == 0 and y == 0:
            print('WARNING! side by dir x = 0 and y = 0')
            return 'c'

        tg = (y / x) if x != 0 else y * float('inf')
        ctg = 1 / tg

        if abs(tg) <= math.tan(pi / 8):
            if x > 0:
                return 'r'
            else:
                return 'l'

        if abs(ctg) <= 1 / math.tan(pi / 8):
            if y > 0:
                return 't'
            else:
                return 'b'

        if math.tan(pi / 8) <= tg <= math.tan(3 * pi / 8):
            if x > 0:
                return 'rt'
            else:
                return 'lb'

        if -math.tan(3 * pi / 8) <= tg <= -math.tan(pi / 8):
            if x > 0:
                return 'rb'
            else:
                return 'lt'

        print('WARNING! Invalid side by dir')
        return 'c'

    def farm(self, resource, farm_side):
        hint_height, hint_width = resource.hint_img.shape

        search_positions = {
            'lt' : Point(720 - 60, 400 - 60),
            't'  : Point(720, 400 - 80),
            'rt' : Point(720 + 60, 400 - 60),
            'l'  : Point(720 - 60, 400),
            'c'  : Point(720, 400 + 20),
            'r'  : Point(720 + 60, 400),
            'lb' : Point(720 - 60, 400 + 60),
            'b'  : Point(720, 400 + 70),
            'rb' : Point(720 + 60, 400 + 60),
        }

        search_priorities = {
            'lt' : ('lt', 't', 'l', 'c', 'rt', 'lb', 'r', 'b', 'rb'),
            't'  : ('t', 'lt', 'rt', 'l', 'r', 'c', 'b', 'lb', 'rb'),
            'rt' : ('rt', 't', 'r', 'c', 'lt', 'rb', 'l', 'b', 'lb'),
            'l'  : ('l', 'lb', 'lt', 'b', 't', 'c', 'r', 'rb', 'rt'),
            'c'  : ('c', 'l', 't', 'r', 'b', 'lt', 'rt', 'rb', 'lb'),
            'r'  : ('r', 'rb', 'rt', 'b', 't', 'c', 'l', 'lb', 'lt'),
            'lb' : ('lb', 'b', 'l', 'c', 'rb', 'lt', 'r', 't', 'rt'),
            'b'  : ('b', 'c', 'lb', 'rb', 'l', 'r', 't', 'lt', 'rt'),
            'rb' : ('rb', 'b', 'r', 'c', 'lb', 'rt', 'l', 't', 'lt'),
        }

        for next_side in search_priorities[farm_side]:
            cursor_pos = search_positions[next_side]
            mousemove(cursor_pos)

            # Waiting for hint
            time.sleep(0.9)

            raw_hint = screenshot(
                cursor_pos + Resource.HINT_OFFSET,
                hint_width, hint_height
            )
            processed_hint = Resource.process_hint(raw_hint)

            if not resource.hint_matches(processed_hint):
                continue

            mouseclick(cursor_pos)

            # Waiting for action start
            time.sleep(1)

            self.wait_action(resource.farm_img)
            break

    def trigger_map(self):
        CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, keyCodeMap['n'], True))
        time.sleep(0.001)
        CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, keyCodeMap['n'], False))
        time.sleep(0.001)

        self.map_opened = not self.map_opened

    def trigger_mount(self):
        CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, keyCodeMap['a'], True))
        time.sleep(0.001)
        CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, keyCodeMap['a'], False))
        time.sleep(0.5)

        self.mounted = not self.mounted

    def mount(self):
        if not self.mounted:
            self.trigger_mount()
            time.sleep(4)

    def unmount(self):
        if self.mounted:
            self.trigger_mount()

    def open_map(self):
        if not self.map_opened:
            self.trigger_map()
            time.sleep(0.8)

    def close_map(self):
        if self.map_opened:
            self.trigger_map()
            time.sleep(0.4)

    def init_current_pos(self):
        self.open_map()
        full_map = Map.full()
        self.close_map()
        self.player_pos = full_map.find_player()

    def init_local_pos(self):
        offset, near_map = Map.around(self.player_pos)
        local_player_pos = near_map.find_player()
        self.player_pos = offset + local_player_pos

    def get_mouse_rel_pos(self, move_direction, dist=120):
        return self.center + move_direction.normalized() * dist

    def get_map(self, wait=DEFAULT_WAIT):
        time.sleep(wait)
        self.map = Map.full()
        self.player_pos = self.map.find_player()

    def add_waypoint(self):
        time.sleep(self.DEFAULT_WAIT)
        self.open_map()
        full_map = Map.full()
        self.close_map()
        player_pos = full_map.find_player()
        self.points.append(WayPoint(player_pos))

    def add_farmpoint(self, resource):
        time.sleep(self.DEFAULT_WAIT)
        self.open_map()
        full_map = Map.full()
        self.close_map()
        player_pos = full_map.find_player()
        self.points.append(FarmPoint(player_pos, resource))

    def get_next_point(self):
        if self.points_iterator is None:
            self.points_iterator = iter(self.points)

        while True:
            try:
                yield next(self.points_iterator)
            except StopIteration:
                self.points_iterator = iter(self.points)
                yield next(self.points_iterator)

    def process_waypoint(self, point):
        move_direction = point.pos - self.player_pos
        self.last_pos = self.player_pos
        pos_counter = 50

        while move_direction.l2() > 12:
            self.init_local_pos()

            if self.player_pos == self.last_pos:
                pos_counter -= 1
            else:
                pos_counter = 50

            if pos_counter == 0:
                raise Exception('Stuck!')

            self.last_pos = self.player_pos

            move_direction = point.pos - self.player_pos
            mouse_pos = self.get_mouse_rel_pos(move_direction, 120)

            mousemove(mouse_pos)
            time.sleep(0.1)

    def process_farmpoint(self, point):
        move_direction = point.pos - self.player_pos
        self.last_pos = self.player_pos
        pos_counter = 50

        while move_direction.l2() > 22:
            self.init_local_pos()

            if self.player_pos == self.last_pos:
                pos_counter -= 1
            else:
                pos_counter = 50

            if pos_counter == 0:
                raise Exception('Stuck!')

            self.last_pos = self.player_pos

            move_direction = point.pos - self.player_pos
            mouse_pos = self.get_mouse_rel_pos(move_direction, 120)

            mousemove(mouse_pos)
            time.sleep(0.1)

        mousemove(Point(720, 400))
        mouserelease(Point(720, 400))
        time.sleep(0.05)

        self.init_local_pos()
        self.unmount()
        self.close_map()

        mousepress(Point(720, 400))
        time.sleep(0.5)
        self.open_map()

        while move_direction.l2() > 6:
            self.init_local_pos()

            if self.player_pos == self.last_pos:
                pos_counter -= 1
            else:
                pos_counter = 50

            if pos_counter == 0:
                raise Exception('Stuck!')

            self.last_pos = self.player_pos

            move_direction = point.pos - self.player_pos
            mouse_pos = self.get_mouse_rel_pos(move_direction, 40)

            mousemove(mouse_pos)
            time.sleep(0.1)

        mouserelease(mouse_pos)
        self.close_map()
#                     time.sleep(0.5)
        self.farm(point.resource, self.get_side_by_dir(move_direction))
        self.mount()
        mousemove(Point(720, 400))
        mousepress(Point(720, 400))
        time.sleep(0.5)
        self.open_map()

    def test(self, wait=DEFAULT_WAIT):
        time.sleep(wait)
        self.init_current_pos()

        self.map_opened = False
        self.mounted    = True
        self.last_pos = self.player_pos

        mousemove(Point(720, 400))
        mousepress(Point(720, 400))
        time.sleep(0.5)
        self.open_map()

        try:
            for point in self.get_next_point():
                if point.__class__.__name__ == 'FarmPoint':
                    self.process_farmpoint(point)
                else:
                    self.process_waypoint(point)
        except Exception:
            raise
        finally:
            self.close_map()
            mouserelease(Point(720, 400))


# Usage:
automove = AlbionAutoMove()
automove.add_waypoint()
automove.add_farmpoint(Wood3())
time.sleep(2)
automove.test()

# Hint image (name of resource near cursor) processing:
time.sleep(4)

hint_height, hint_width = resource.hint_img.shape
raw_hint = screenshot(
    mouseposition() + Resource.HINT_OFFSET,
    hint_width, hint_height
)
processed_hint = Resource.process_hint(raw_hint)

# Processed hint plot:
plt.figure(figsize=(10, 10))
plt.imshow(processed_hint, cmap=plt.cm.gray)
plt.grid()
plt.xticks(np.arange(0, processed_hint.shape[1], 2))
plt.yticks(np.arange(0, processed_hint.shape[0], 2))
plt.show()

# Processed hint export:
with open('resources/wood_2_hint.bin', 'wb') as f:
    f.write((processed_hint*255).astype(np.uint8).tobytes())


# Farm image processing:
time.sleep(6)
wood_img = screenshot(
    Resource.FARM_OFFSET,
    32, 32
)

# Farm image plot:
plt.imshow(wood_img)
plt.show()

# Farm image export:
with open('resources/wood_3_farm.bin', 'wb') as f:
        f.write(wood_img.tobytes())


# Image properties load:
with open('resources/properties.json', 'r') as f:
    properties = json.load(f)

# Image properties dump:
with open('resources/properties.json', 'w') as f:
    json.dump(properties, f)

"""
graphics_utils.py
----------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""


import os.path
import sys
import time
import tkinter as tk

WINDOWS = sys.platform == 'win32'  # True if on Win95/98/NT

__root_window__ = None  # The root window for graphics output
__canvas__ = None  # The canvas which holds graphics
__canvas_xs__ = None  # Size of canvas this_ojb
__canvas_ys__ = None
__canvas_x__ = None  # Current position on canvas
__canvas_y__ = None
__canvas_col__ = None  # Current colour (set to black below)
__bg_color__ = None

__left_click_loc__ = None
__right_click_loc__ = None
__ctrl_left_click_loc__ = None
__mouse_enabled__ = 0

CANVAS_T_SIZE = 12
CANVAS_T_SERIFS = 0


def format_color(r, g, b):
    """
    :param r:
    :param g:
    :param b:
    :return:
    """
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))


def color_to_vector(color):
    """
    :param color:
    :return:
    """
    return [int(x, 16) / 256.0 for x in [color[1:3], color[3:5], color[5:7]]]


if WINDOWS:
    _canvas_tfonts = ['times new roman', 'lucida console']
else:
    _canvas_tfonts = ['times', 'lucidasans-24']
    # pass  # XXX need defaults here


def sleep(secs):
    """
    :param secs:
    :return:
    """
    global __root_window__
    if __root_window__ is None:
        time.sleep(secs)
    else:
        __root_window__.update_idletasks()
        __root_window__.after(int(1000 * secs), __root_window__.quit)
        __root_window__.mainloop()


def begin_graphics(width=640, height=480, color=format_color(0, 0, 0), title=None):
    """
    :param width:
    :param height:
    :param color:
    :param title:
    :return:
    """
    global __root_window__, __canvas__, __canvas_x__, __canvas_y__, \
        __canvas_xs__, __canvas_ys__, __bg_color__

    # Check for duplicate call
    if __root_window__ is not None:
        # Lose the window.
        __root_window__.destroy()

    # Save the canvas size parameters
    __canvas_xs__, __canvas_ys__ = width - 1, height - 1
    __canvas_x__, __canvas_y__ = 0, __canvas_ys__
    __bg_color__ = color

    # Create the root window
    __root_window__ = tk.Tk()
    __root_window__.protocol('WM_DELETE_WINDOW', _destroy_window)
    __root_window__.title(title or 'Graphics Window')
    __root_window__.resizable(0, 0)

    # Create the canvas this_ojb
    try:
        __canvas__ = tk.Canvas(__root_window__, width=width, height=height)
        __canvas__.pack()
        draw_background()
        __canvas__.update()
    except:
        __root_window__ = None
        raise

    # Bind to key-down and key-up events
    __root_window__.bind("<KeyPress>", _keypress)
    __root_window__.bind("<KeyRelease>", _keyrelease)
    __root_window__.bind("<FocusIn>", _clear_keys)
    __root_window__.bind("<FocusOut>", _clear_keys)
    __root_window__.bind("<Button-1>", _leftclick)
    __root_window__.bind("<Button-2>", _rightclick)
    __root_window__.bind("<Button-3>", _rightclick)
    __root_window__.bind("<Control-Button-1>", _ctrl_leftclick)
    _clear_keys()


def _leftclick(event):
    """
    :param event:
    :return:
    """
    global __left_click_loc__
    __left_click_loc__ = (event.x, event.y)


def _rightclick(event):
    """
    :param event:
    :return:
    """
    global __right_click_loc__
    __right_click_loc__ = (event.x, event.y)


def _ctrl_leftclick(event):
    """
    :param event:
    :return:
    """
    global __ctrl_left_click_loc__
    __ctrl_left_click_loc__ = (event.x, event.y)


def wait_for_click():
    """
    :return:
    """
    while True:
        global __left_click_loc__
        global __right_click_loc__
        global __ctrl_left_click_loc__
        if __left_click_loc__ is not None:
            val = __left_click_loc__
            __left_click_loc__ = None
            return val, 'left'
        if __right_click_loc__ is not None:
            val = __right_click_loc__
            __right_click_loc__ = None
            return val, 'right'
        if __ctrl_left_click_loc__ is not None:
            val = __ctrl_left_click_loc__
            __ctrl_left_click_loc__ = None
            return val, 'ctrl_left'
        sleep(0.05)


def draw_background():
    """
    :return:
    """
    corners = [(0, 0), (0, __canvas_ys__), (__canvas_xs__, __canvas_ys__), (__canvas_xs__, 0)]
    polygon(corners, __bg_color__, fill_color=__bg_color__, filled=True, smoothed=False)


def _destroy_window( ):
    """
    :return:
    """
    sys.exit(0)


#    global ROOT_WINDOW
#    ROOT_WINDOW.destroy()
#    ROOT_WINDOW = None
# print "DESTROY"

def end_graphics():
    """
    :return:
    """
    global __root_window__, __canvas__, __mouse_enabled__
    try:
        try:
            sleep(1)
            if __root_window__ is not None:
                __root_window__.destroy()
        except SystemExit as e:
            print('Ending graphics raised an exception:', e)
    finally:
        __root_window__ = None
        __canvas__ = None
        __mouse_enabled__ = 0
        _clear_keys()


def clear_screen(background=None):
    """
    :param background:
    :return:
    """
    global __canvas_x__, __canvas_y__
    __canvas__.delete('all')
    draw_background()
    __canvas_x__, __canvas_y__ = 0, __canvas_ys__


def polygon(coords, outline_color, fill_color=None, filled=1, smoothed=1, behind=0, width=1):
    """
    :param coords:
    :param outline_color:
    :param fill_color:
    :param filled:
    :param smoothed:
    :param behind:
    :param width:
    :return:
    """
    c = []
    for coord in coords:
        c.append(coord[0])
        c.append(coord[1])
    if fill_color is None:
        fill_color = outline_color
    if filled == 0:
        fill_color = ''
    poly = __canvas__.create_polygon(c,
                                     outline=outline_color,
                                     fill=fill_color,
                                     smooth=smoothed,
                                     width=width)
    if behind > 0:
        __canvas__.tag_lower(poly, behind)  # Higher should be more visible
    return poly


def square(pos, r, color, filled=1, behind=0):
    x, y = pos
    coords = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    return polygon(coords, color, color, filled, 0, behind=behind)


def circle(pos, r, outline_color, fill_color, endpoints=None, style='pieslice', width=2):
    x, y = pos
    x0, x1 = x - r - 1, x + r
    y0, y1 = y - r - 1, y + r
    if endpoints is None:
        e = [0, 359]
    else:
        e = list(endpoints)
    while e[0] > e[1]:
        e[1] = e[1] + 360

    return __canvas__.create_arc(x0, y0, x1, y1, outline=outline_color, fill=fill_color,
                                 extent=e[1] - e[0], start=e[0], style=style, width=width)


def image(pos, file="../../blueghost.gif"):
    x, y = pos
    # img = PhotoImage(file=file)
    return __canvas__.create_image(x, y, image=tk.PhotoImage(file=file), anchor=tk.NW)


def refresh():
    __canvas__.update_idletasks()


def move_circle(my_id, pos, r, endpoints=None):
    global __canvas_x__, __canvas_y__

    x, y = pos
    #    x0, x1 = x - r, x + r + 1
    #    y0, y1 = y - r, y + r + 1
    x0, x1 = x - r - 1, x + r
    y0, y1 = y - r - 1, y + r
    if endpoints is None:
        e = [0, 359]
    else:
        e = list(endpoints)
    while e[0] > e[1]:
        e[1] = e[1] + 360

    if os.path.isfile('flag'):
        edit(my_id, ('extent', e[1] - e[0]))
    else:
        edit(my_id, ('start', e[0]), ('extent', e[1] - e[0]))
    move_to(my_id, x0, y0)


def edit(my_id, *args):
    __canvas__.itemconfigure(my_id, **dict(args))


def text_on_canvas(pos, color, contents, font='Helvetica', size=12, style='normal', anchor="nw"):
    global __canvas_x__, __canvas_y__
    x, y = pos
    font = (font, str(size), style)
    return __canvas__.create_text(x, y, fill=color, text=contents, font=font, anchor=anchor)


def change_text(id, new_text, font=None, size=12, style='normal'):
    __canvas__.itemconfigure(id, text=new_text)
    if font is not None:
        __canvas__.itemconfigure(id, font=(font, '-%d' % size, style))


def change_color(id, new_color):
    __canvas__.itemconfigure(id, fill=new_color)


def line(here, there, color=format_color(0, 0, 0), width=2):
    x0, y0 = here[0], here[1]
    x1, y1 = there[0], there[1]
    return __canvas__.create_line(x0, y0, x1, y1, fill=color, width=width)


##############################################################################
### Keypress handling ########################################################
##############################################################################

# We bind to key-down and key-up events.

KEYS_DOWN = {}
KEYS_WAITING = {}
# This holds an unprocessed key release.  We delay key releases by up to
# one call to keys_pressed() to get round a problem with auto repeat.
GOT_RELEASE = None


def _keypress(event):
    global GOT_RELEASE
    # remap_arrows(event)
    KEYS_DOWN[event.keysym] = 1
    KEYS_WAITING[event.keysym] = 1
    #    print event.char, event.keycode
    GOT_RELEASE = None


def _keyrelease(event):
    global GOT_RELEASE
    # remap_arrows(event)
    try:
        del KEYS_DOWN[event.keysym]
    except:
        pass
    GOT_RELEASE = 1


def remap_arrows(event):
    # TURN ARROW PRESSES INTO LETTERS (SHOULD BE IN KEYBOARD AGENT)
    if event.char in ['a', 's', 'd', 'w']:
        return
    if event.keycode in [37, 101]:  # LEFT ARROW (_win / x)
        event.char = 'a'
    if event.keycode in [38, 99]:  # UP ARROW
        event.char = 'w'
    if event.keycode in [39, 102]:  # RIGHT ARROW
        event.char = 'd'
    if event.keycode in [40, 104]:  # DOWN ARROW
        event.char = 's'


def _clear_keys(event=None):
    global KEYS_DOWN, GOT_RELEASE, KEYS_WAITING
    KEYS_DOWN = {}
    KEYS_WAITING = {}
    GOT_RELEASE = None


def keys_pressed(d_o_e=None, d_w=None):
    if d_o_e is not None:
        d_o_e(d_w)
        if GOT_RELEASE:
            d_o_e(d_w)
    return list(KEYS_DOWN.keys())


def keys_waiting():
    global KEYS_WAITING
    keys = list(KEYS_WAITING.keys())
    KEYS_WAITING = {}
    return keys


# Block for a list of keys...

def wait_for_keys():
    keys = []
    while not keys:
        keys = keys_pressed()
        sleep(0.05)
    return keys


def remove_from_screen(x, d_o_e=None, d_w=None):
    __canvas__.delete(x)
    if d_o_e is not None:
        d_o_e(d_w)


def _adjust_coords(coord_list, x, y):
    for i in range(0, len(coord_list), 2):
        coord_list[i] = coord_list[i] + x
        coord_list[i + 1] = coord_list[i + 1] + y
    return coord_list


def move_to(this_ojb, x, y=None, d_o_e=None, d_w=None):
    if y is None:
        try:
            x, y = x
        except:
            raise Exception('incomprehensible coordinates')

    horiz = True
    new_coords = []
    current_x, current_y = __canvas__.coords(this_ojb)[0:2]  # first point
    for coord in __canvas__.coords(this_ojb):
        if horiz:
            inc = x - current_x
        else:
            inc = y - current_y
        horiz = not horiz

        new_coords.append(coord + inc)

    __canvas__.coords(this_ojb, *new_coords)
    if d_o_e is not None:
        d_o_e(d_w)


def move_by(object, x, y=None,
            d_o_e=lambda arg: __root_window__.dooneevent(arg),
            d_w=tk._tkinter.DONT_WAIT, lift=False):
    if y is None:
        try:
            x, y = x
        except:
            raise Exception('incomprehensible coordinates')

    horiz = True
    new_coords = []
    for coord in __canvas__.coords(object):
        if horiz:
            inc = x
        else:
            inc = y
        horiz = not horiz

        new_coords.append(coord + inc)

    __canvas__.coords(object, *new_coords)
    d_o_e(d_w)
    if lift:
        __canvas__.tag_raise(object)


def write_postscript(filename):
    """Writes the current canvas to a postscript file."""
    psfile = open(filename, 'w')
    psfile.write(__canvas__.postscript(pageanchor='sw',
                                       y='0.c',
                                       x='0.c'))
    psfile.close()


ghost_shape = [
    (0, - 0.5),
    (0.25, - 0.75),
    (0.5, - 0.5),
    (0.75, - 0.75),
    (0.75, 0.5),
    (0.5, 0.75),
    (- 0.5, 0.75),
    (- 0.75, 0.5),
    (- 0.75, - 0.75),
    (- 0.5, - 0.5),
    (- 0.25, - 0.75)
]

if __name__ == '__main__':
    begin_graphics()
    clear_screen()
    ghost_shape = [(x * 10 + 20, y * 10 + 20) for x, y in ghost_shape]
    g = polygon(ghost_shape, format_color(1, 1, 1))
    move_to(g, (50, 50))
    circle((150, 150), 20, format_color(0.7, 0.3, 0.0), format_color(0, 0, 0), endpoints=[15, - 15])
    sleep(2)

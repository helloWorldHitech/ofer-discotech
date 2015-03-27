# This has all the constants and hardcoded magic numbers used.

muteDebugPrints = False

# for keyboard calibration: which key corresponds to which state
# for image display: short name for each state and location to display psr
positionKeyAndName = {
#   key : [ 'stateName', 'shortName', 'displayPosition' ]
    's' : [ 'center',    'C',         0                 ],
    'a' : [ 'left',      'L',         25                ],
    'd' : [ 'right',     'R',         50                ],
    'w' : [ 'up',        'U',         75                ],
    'x' : [ 'down',      'D',         100               ],
    'z' : [ 'closed',    '-',         125               ],
}

# which state to revert to if nothing is identified
defaultState = 'nothing'

# which state is the reference state
referenceState = 'center'

# psrs state to refState ratio - by how much a state psr should be above the reference state
# to trigger an identification
psrRefRatio = 1.3

# properties of the states: how long it takes to enter, exist, broadcast, what to broadcast etc.
stateKeyAndProperties = {
#   key      : [ timeToIn, timeToOut, timeToBroadcast, broadcast, defaultOout, ]
    'center' : [ 0.125,    0.125,     None,            None,      defaultState,   ],
    'left'   : [ 0.125,    0.125,     0.2,             'L',       defaultState,   ],
    'right'  : [ 0.125,    0.125,     0.2,             'R',       defaultState,   ],
    'up'     : [ 0.125,    0.125,     0.2,             'U',       defaultState,   ],
    'down'   : [ 0.125,    0.125,     0.2,             'D',       defaultState,   ],
    'closed' : [ 0.125,    0.125,     0.25,            'B',       defaultState,   ],
    'nothing': [ 0.125,    0.125,     None,            None,      defaultState,   ],
}  

# how much to resize the image to work on (for speed's sake)
imageSizeRatio = 0.6

# which portion of the image to take for initial window pick. window will be portion*width X portion*height
initialWindowPortion = 0.6

# number of random affine movements to create when making a filter from single image
numMoves = 128


# mp3 files dir (relative to the run directory

mp3Dir = 'mp3'

# how much to wait between the 'please look up/down/... now' and actually snapping
captureTimeDelay = 1.0

mp3FileMap = {
    "calibration introduction" : "calibration_intro.mp3",
    "que"                      : "now.mp3",
    "ack"                      : "minor_ack.mp3",
    "final ack"                : "major_ack.mp3",
    "instruction"              : {
        "up"                   : "please_look_up.mp3",
        "down"                 : "please_look_down.mp3",
        "left"                 : "please_look_left.mp3",
        "right"                : "please_look_right.mp3",
        "closed"               : "please_close_your_eyes.mp3",
        "center"               : "please_look_straight.mp3",
        },
    "recognition"              : {
        "U"                   : "up.mp3",
        "D"                   : "down.mp3",
        "L"                   : "left.mp3",
        "R"                   : "right.mp3",
        "B"                   : "closed.mp3",
    }
}

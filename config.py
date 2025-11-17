# =========================
# General configuration
# =========================
BACKBONE = "ViT-L-14"           # "ViT-B-16" (fast) or "ViT-L-14" (bigger)
PRETRAINED = "openai"           # open_clip weights tag

# Zero-shot class prompts (edit/extend as needed)
CLASSES = {
    "ASCEND": [
        "the ascend gesture is demonstrated by a diver giving a thumbs-up sign, with the thumb pointing upwards",
        "diver with one arm up in front of the cheast, upper-body gesture"
    ],
    "FOLLOWME": [
    "the diver performs this gesture using only one hand, which can be either the left or right hand, raised in front of the body",
    "the palm of the hand is oriented toward the diver, with fingers tightly together",
    "the forearm repeatedly bends inward toward the upper arm and then extends outward again, creating a back-and-forth curling motion that signals 'follow me'"
    ],
    "LEFT": [
        "underwater diver with left arm extended horizontally to the left",
        "a diver pointing left with the left arm straight, elbow extended"
    ],
    "RIGHT": [
        "underwater diver with right arm extended horizontally to the right",
        "a diver pointing right with the right arm straight, elbow extended"
    ],
    "STOP": [
    "underwater diver with one hand raised in front of the body, palm facing outward",
    "a diver signaling stop with the arm bent at the elbow and fingers extended together"
    ],
    "DESCEND": [
    "the descend gesture is demonstrated by a diver raising one hand in front of the chest and forming a thumbs-down sign",
    "the thumb is extended downward while the fingers remain closed in a fist, signaling the motion to descend underwater; can be performed with either left or right hand"
    ],
    "BUDDYUP": [
    "underwater diver holds both hands in front of the chest, index fingers pointing outward horizontally",
    "the diver moves both hands closer together until the index fingers nearly touch, then separates them outward again",
    "this closing and opening motion is repeated several times to signal body up"
    ],
    "LEVEL": [
    "underwater diver holds one hand flat and horizontal in front of the chest, palm facing downward",
    "fingers are kept together with the hand steady, signaling level; can be performed with either left or right hand"
    ],
    "ME": [
    "underwater diver raises one hand and uses the extended index finger to point inward toward the chest",
    "the pointing motion is directed at the diver’s own chest to indicate 'me'; can be performed with either left or right hand"
    ],
    "OKAY": [
    "underwater diver raises one hand in front of the chest, forming a circle by touching the tip of the thumb and index finger together",
    "the remaining three fingers are extended upward while the circular shape of the thumb and index finger clearly signals 'OK'; can be performed with either left or right hand"
    ],
    "YOU": [
    "underwater diver raises one hand in front of the body, extending the index finger to point outward",
    "the diver keeps the other fingers curled while the index finger is directed straight ahead, signaling the YOU gesture; can be performed with either left or right hand"
    ],
    "NONE": [
        "underwater diver not gesturing, casual movements, no command"
    ],
}

# Thresholds / smoothing
TAU_OTHER = 0.27        # reject to 'Other' if max sim below this
SMOOTH_WINDOW = 21      # temporal window length for smoother
SMOOTH_PERSIST = 6      # frames needed before switching to new class
SMOOTH_CONF = 0.55      # smoother-level rejection to 'Other'

# Video sampling
SAMPLE_K = 60           # frames sampled per clip
YOLO_PAD = 0.25         # extra padding around person bbox
USE_YOLO = True         # set False to skip detection and use full frame

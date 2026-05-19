BLUR_SHARP = 0
BLUR_MOTION = 1
BLUR_DEFOCUS = 2

BLUR_NAME_TO_ID = {
    "sharp": BLUR_SHARP,
    "none": BLUR_SHARP,
    "motion": BLUR_MOTION,
    "motion_blur": BLUR_MOTION,
    "defocus": BLUR_DEFOCUS,
    "defocus_blur": BLUR_DEFOCUS,
}


def normalize_blur_type(value):
    if isinstance(value, int):
        if value in (BLUR_SHARP, BLUR_MOTION, BLUR_DEFOCUS):
            return value
        return BLUR_MOTION
    name = str(value or "motion").strip().lower()
    return BLUR_NAME_TO_ID.get(name, BLUR_MOTION)

def enhance_prompt(prompt: str, platform: str) -> str:
    platform_styles = {
        "tiktok": "vertical format, trending, viral, dynamic movement, engaging, colorful, fast-paced",
        "youtube": "cinematic, high quality, professional, smooth camera movement, detailed",
    }
    enhanced = f"{prompt}, {platform_styles.get(platform, '')}, high quality, 4k, smooth animation, professional lighting"
    return enhanced[:197] + "..." if len(enhanced) > 200 else enhanced

# 
def get_negative_prompt(platform: str) -> str:
    base_negative = "blurry, low quality, distorted, ugly, glitchy, stuttering, static"
    return f"{base_negative}, boring, slow motion, text, watermark" if platform == "tiktok" else f"{base_negative}, amateur, shaky camera, poor lighting"

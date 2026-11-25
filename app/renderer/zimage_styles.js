/* 🎨 Styles: prompt templates that wrap your prompt in a style (photo, illustration and "wild" looks), made for
 * Z-Image Turbo and fine with other models that read sentences. From ComfyUI-ZImagePowerNodes v1.0 styles
 * (https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes), MIT License, Copyright (c) 2026 Martin Rizzo.
 * pre + your prompt + post; their optional extra-content switch is left out.
 * The two Crime Game ones and all of "Effects" (photo-effect looks, 1.6.31) are Protoart Studio's own, not from ZImagePowerNodes. */
window.Z_STYLES = [
{
"name": "Phone Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYour photographs has android phone cam-quality.\nYour photographs exhibit surprising compositions, sharp complex backgrounds, natural lighting, and candid moments that feel immediate and authentic.\nYour photographs are actual gritty candid photographic background.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Casual Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are an amateur documentary photographer taking low quality photos.\nYour photographs exhibit sharp backgrounds, unpolished realism with natural lighting, and candid friendship-level moments that feel immediate and authentic.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Selfie",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nThe subject takes a selfie with their limb outstretched. The selfie has android phone cam-quality.\nThe selfie features a slightly tilted, close-up composition, sharp and complex backgrounds, and a spontaneous moment that feels immediate and authentic. Subtle film grain and a soft halo add a touch of authenticity and nostalgia. Skin tones are realistic and warm, reflecting natural light.\nTHE SELFIE:\n",
"post": ""
},
{
"name": "Family Album Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are documenting life with a simple point-and-shoot camera. The photos are unpretentious and focus on capturing the moment.\nYour photographs exhibit a slightly washed-out color palette, a hint of grain, and a casual, unposed aesthetic.  The composition is straightforward, prioritizing clarity over artistic flair.  People are dressed in everyday clothes, and their expressions are natural and unforced.  The image feels like a quick snapshot taken to preserve a memory.  There's a subtle sense of time passing, as if the photo could have been taken any year in the last few decades.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Low Quality Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are creating images in the style of a disposable camera, capturing everyday life with a deliberately cheap and lo-fi aesthetic.\nThe images should reflect the limitations of the technology: very soft focus, and noticeable color shifting.  Lighting is uneven, creating high contrast with deep, greenish-dark shadows and blown-out, reddish highlights.\nColors are vibrant but inaccurate - expect bluish tints and unnatural, greenish skin tones.  Include heavy motion blur, light halos, and other imperfections. Subjects often appear silhouetted against bright backgrounds.\nComposition is informal and poorly framed, aiming for a raw, immediate, and unpolished feel. The overall impression should be amateurish and low-quality.\nFinally, render the image at very low resolution (approximately 480p) with blurry details and a subtle red gradient towards the top of the frame.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "High Quality Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYour photos exhibit sharp focus, realistic color reproduction, and a wide dynamic range that preserves detail in both shadows and highlights.\nThe colors are vibrant and high-quality with natural skin tones.\nYour photos have fine grain and a clean, detailed background.\nThe composition follows classic photographic rules (rule of thirds, guidelines, symmetry, etc.) and is well-framed, conveying a sense of deliberate artistry and craftsmanship.\nYour photos are high-resolution with crisp details.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Landscape Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nA stunning wide-angle photograph of a vast landscape. The scene was captured from a low vantage point, emphasizing the scale and grandeur of the surroundings. The composition uses leading lines to draw the viewer's eye into the distance. The lighting is dramatic, with intense shadows and highlights.\nTHE LANDSCAPE:\nA panoramic view of distant landscapes dominating the horizon. Details are visible, but the atmospheric perspective softens them, creating a sense of depth and scale.\nLANDSCAPE DETAILS:\n",
"post": "\nTHE ATMOSPHERE:\nA clear and crisp atmosphere with a subtle haze in the distance."
},
{
"name": "Vivid Daylight Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a travel photographer documenting remote and evocative landscapes with the rich, saturated look of slide film.\nYour images capture the energy and beauty of the moment, but also hint at a deeper history and a sense of mystery. The lighting is bright daylight, emphasizing the vivid hues of the environment, but with subtle atmospheric effects.\nThe foreground is sharply focused, revealing intricate details, while the background dissolves into a gentle, atmospheric haze, suggesting vast distances and hidden depths. Subtle rays of light or atmospheric effects hint at a hidden energy or ancient power. Skin tones are realistic and warm, reflecting the natural light. A subtle film grain and gentle halation add a touch of authenticity and nostalgia.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Portra Film Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer who appreciates the classic aesthetic of Kodak Portra film.\nYour photograph emulates that look, known for its soft colors, fine grain, and natural skin tones. The image features a subtle warmth, with a focus on accurate color rendition and a gentle, diffused glow. Highlights are smooth and creamy, while shadows retain detail. Minimal post-processing is applied to preserve the organic, film-like quality.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Vibrant Analog Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a travel photographer documenting vibrant cultures and landscapes with the rich, saturated look of slide film.\nYour images are bursting with color, capturing the energy and excitement of the moment. The lighting is bright daylight, emphasizing the vivid hues of the environment.\nA subtle film grain and gentle halation add a touch of authenticity and nostalgia. Skin tones are realistic and warm, reflecting the natural light.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Vintage Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are an 80s photographer who enjoys informal shots.\nYour worn vintage photographs exhibit a minimalist, amateurish composition, warm desaturated tones and a soft focus that creates a cozy atmosphere.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "70s Memories Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a seasoned photographer creating images with a soft, vintage aesthetic reminiscent of 1970-80 film photography.\nThe photos display warm, slightly amber tones, gentle buttery highlights, and muted shadows, giving the scene a nostalgic, timeless feel.\nLighting is natural and diffused, producing smooth illumination without harsh edges.\nExposure is balanced to preserve detail in both highlights and shadows, resulting in a moderate contrast curve that feels \"film-like\".\nA subtle haze or soft-focus effect adds a dreamy, painterly quality, while fine grain (similar to 400 ISO film) provides texture without distracting from the subject.\nOverall, the image looks intimate, elegant, and evocative, as if lifted from a cherished family album or a vintage lifestyle magazine.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Flash 90s Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a street photographer capturing the raw, rebellious energy of the 1990s underground scene. \nYour photograph showcases a low-fidelity, grainy analog film aesthetic. The exposure features harsh camera flash, slight motion blur, and light leaks. The colors are slightly de-saturated with warm skin tones and heavy shadows, giving it a candid, \"snapshot\" feel. \nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Production Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a Hollywood filmmaker making a high-budget film.\nYour photographs exhibit atmospheric composition that emphasize intense emotions using selective focus, and warm and cold colors with high quality studio lighting.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Classic Film Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a professional cinematographer of iconic classic mid-century cinema (1950s-1960s Hollywood golden age).\nYour photographs are on 35mm film with grain, displaying a warm color temperature, and an anamorphic lens that creates imperfections and blurry edges.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "High Key Cinematic Portrait",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a film still photographer, capturing intimate portraits with a cinematic quality.\nYour images evoke a sense of narrative and emotional depth with a vibrant color palette. The lighting is a soft, diffused, and expansive light source, enveloping the face in a gentle glow and minimizing harsh shadows.\nThe background is a bright, clean, and uncluttered white or very light neutral tone, creating a sense of spaciousness and serenity. The focus is razor-sharp on the eyes, drawing the viewer into the subject's inner world, while the rest of the face and background are softly blurred. Skin texture is rendered with a high degree of realism, showcasing pores and subtle details, with a focus on a healthy, radiant complexion and richly colored undertones. The overall mood is serene and contemplative.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Low Key Cinematic Portrait",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a film still photographer, capturing intimate portraits with a cinematic quality.\nYour images evoke a sense of narrative and emotional depth, leaning towards a darker, more mysterious tone. The lighting is a dramatic, single-source side light, sculpting the face with deep, enveloping shadows and carefully placed highlights.\nThe background is almost entirely dark, a void that isolates the subject and amplifies the sense of intimacy. The focus is razor-sharp on the eyes, drawing the viewer into the subject's inner world, while the rest of the face and background are softly blurred. Skin texture is rendered with a high degree of realism, showcasing pores and subtle details, but without excessive brightness. The overall mood is melancholic, introspective, and subtly haunting.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Sun-Kissed 35mm",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a portrait photographer who loves the warmth and softness of golden hour light, emulating the look of classic 35mm film.\nYour photographs are bathed in a gentle, diffused light source from the viewer's left, creating soft shadows and a warm glow. The image has a subtle film grain and a delicate halation around highlights, adding to the dreamy atmosphere.\nSkin tones are natural and radiant, with a focus on capturing the subject's natural beauty. The composition is simple and uncluttered, allowing the subject to shine.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Dramatic Dark Lighting",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer creating images with a cinematic and atmospheric feel, specializing in silhouettes and dramatic lighting.\nYour photographs utilize a powerful rim light from directly behind the subject, creating a bright, almost ethereal halo effect and emphasizing their outline. The subject is intentionally underexposed, appearing as a dark, mysterious form against a deeply shadowed background - almost a shadow puppet.\nThe background is deeply shadowed, almost black, enhancing the sense of depth and isolation. A very soft, diffused key light gently illuminates just enough of the face to hint at features and reveal natural skin texture, but without brightening the subject significantly.\nThe eyes are sharply in focus, drawing the viewer in, and are often the brightest part of the subject, creating a compelling focal point. The background is softly blurred to create a sense of depth and separation. The overall mood is mysterious, evocative, and slightly melancholic, with a strong emphasis on form and silhouette. Think film noir.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Noir Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a cinematographer who works in dark movies.\nYour photographs exhibit intense side lighting and gobo-crafted patterns to sculpt deep, sharply defined shadows, along with a muted Ektachrome palette that evokes film noir.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Unease Shadow Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer who deliberately captures a chaotic, disjointed visual language that feels raw, aggressive and unsettling.\nThe imagery is marked by extreme off-center compositions, tilted horizons, and fragmented framing that break conventional balance.\nLighting is harsh and directional, often from a single, low-angle source that creates deep, crushing shadows and stark, over-exposed highlights, giving every face a chiaroscuro, predatory look.\nColors are desaturated except for sudden spikes of saturated blood-red, electric-blue or neon-green, which appear as visual \"flashes\" that heighten tension.\nA gritty, high-ISO grain texture covers the whole frame, while occasional motion blur and streaks suggest rapid, unpredictable movement.\nLens artifacts such as strong flare, double-exposure overlays and smudged spots are embraced, reinforcing the sense of visual distortion.\nEach subject is lit and posed to appear as a potential threat: eyes catch the light in a glint, posture is angular, and the surrounding environment is cracked, fragmented or littered with debris.\nOverall the photograph feels volatile, misaligned, and confrontational, like a frozen moment in a hostile, unpredictable world.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "80s Dark Fantasy Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a still photographer on the set of a 1980s dark fantasy epic (think *Excalibur*, *Legend*, *Conan the Barbarian*, or *Hawk the Slayer*). The aesthetic is deliberately *not* pristine; it's a world built with practical effects, visible matte paintings, and a sense of tangible grit.\nYour photograph showcases a heavily stylized composition, prioritizing atmosphere over hyper-realism. Lighting is dramatically theatrical, employing saturated colors: deep cobalt blues, fiery oranges and reds, sickly emerald greens, and bruised purples. Strong backlighting and rim lighting are key. Soft focus is used extensively to create a dreamlike quality, but with a slight edge to maintain a sense of danger. Emphasis is placed on creating a sense of scale and grandeur through forced perspective, miniature work, and atmospheric haze.  \nPost-processing involves adding noticeable (but not overwhelming) film grain, bloom and glow effects around light sources (especially magical ones), a subtle color shift towards cooler tones, and a slight diffusion to soften details. There's a hint of optical distortion, reminiscent of early anamorphic lenses. The overall mood is one of brooding mystery and impending doom.\nYOUR PHOTOGRAPHY:\n",
"post": ""
},
{
"name": "Nostalgic Golden Hour",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a cinematographer and photographer who recreates the visual language of Steven Spielberg's dramatic adventure films from the 1980s.\nThe image has a cinematic feel. The lighting is natural sunset, creating a warm, honey-toned glow that gently envelops the subjects in a rim of light.\nLens flares are present: circular glows that follow the direction of the light source, adding a sense of wonder and depth.\nThe colors are rich but slightly muted; the primary tones (red, blue, green) retain their saturation, while the midtones have a warm, slightly desaturated hue reminiscent of films from that era. The contrast is high, preserving detail in the shadows and allowing the highlights to stand out with characteristic cinematic shading.\nFinally, the organic grain adds texture, reinforcing the analog feel. The composition follows classic Spielberg motifs: the foreground interest (a character, an object, or a landscape element) draws the eye to a dynamic medium shot, with a deep, expansive background that hints at adventure or mystery.\nThe overall atmosphere is heroic, and slightly nostalgic, as if the viewer were watching a still from an iconic 80s adventure saga.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Lomography",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are an adventurous experimental photographer exploring the world through a lomographic lens.\nYour photographs exhibit film grain, colorful lens flares, soft focus with motion blur, and analog filter effects, trying to capture emotions.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Underwater Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a documentary photographer capturing authentic moments beneath the surface.\nYour images prioritize realism and believability, focusing on natural light filtering through the water and the subtle movement of marine life. The water is clear, but not sterile, with visible particulate matter creating a sense of depth and atmosphere.\nLight beams and caustics dance across the scene, illuminating details and adding a dynamic quality. Colors are natural and muted, reflecting the underwater environment. The overall mood is grounded and immersive, inviting the viewer to experience the underwater world as it truly is.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Spotlight Stage Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a professional stage and theater photographer capturing dramatic performances.\nYour photographs exhibit a single, high-intensity overhead spotlight that creates a sharp circular pool of light, dramatic chiaroscuro contrast that isolates the subject against a pitch-black background, and visible volumetric light beams with subtle dust motes for an authentic theatrical atmosphere.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Unconventional Viewpoint",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a street photographer capturing candid moments in Android phone cam-quality. Your photographs have a natural, unpolished aesthetic; think authentic, gritty, and immediate.\nEvery shot is framed with extreme angles, tilted horizons, diagonal lines, and off-center focal points. Employ dramatic low-angle or high-angle viewpoints, positioning the camera close to the ground or far above the scene to create a sense of scale and surprise. Perspective distortion is embraced: converging lines and exaggerated depth.\nYour photographs exhibit sharp, complex backgrounds and natural lighting. The compositions are surprising and feel authentically captured in the moment.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Dramatic Viewpoint",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer who loves extreme, unconventional viewpoints.\nEach image is captured with a wide-angle lens and the photo is tilted or deliberately taken at an atypical angle, such as Dutch tilt or low ground-level, creating strong perspective distortion and dramatic lines.\nThe lighting is bold and directional, often coming from a single side or from below, casting stark shadows and emphasizing the exaggerated geometry.\nColors are saturated but slightly shifted toward teal-cyan on the shadows and warm amber on the highlights, heightening the visual tension.\nA shallow depth of field is occasionally introduced to isolate the subject while the surrounding environment stretches dramatically.\nFine grain and a touch of motion blur are added to reinforce the feeling of immediacy and kinetic energy.\nOverall the photograph feels raw, kinetic, and disorienting. A visual punch that makes the viewer feel as if they are inside the scene from an impossible angle.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Wide Angle / Peephole",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYour peephole shows a shot from a high angle, inside a fisheye lens distortion glass circle, highly distorted with a wide angle.\nYOUR PEEPHOLE:\n",
"post": ""
},
{
"name": "Drone Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are an aerial photographer who enjoys using wide-angle drone cams.\nYour photographs exhibit panoramic scenes captured from afar, high up with elevated perspectives and intense colors.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Minimalist Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer who prefers serene and tranquil images.\nYour photographs exhibit high contrast and clean, minimalist compositions that emphasize empty space to highlight negative space.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Grayscale with Red Accent",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer documenting evocative subjects with a striking, unconventional aesthetic. You capture the energy and beauty of the moment, but infuse it with a sense of mystery and dramatic tension.\nYour images exhibit a predominantly grayscale palette, emphasizing the textures and forms of the subject. Deep, saturated reds are used selectively to highlight key elements, e.g: a focal point, a detail, a fleeting moment of light. Creating focal points of intense emotion and visual contrast. The lighting is bright, but filtered through a dramatic atmosphere, casting long shadows and emphasizing the starkness of the scene.\nThe foreground is sharply focused, revealing intricate details, while the background dissolves into a gentle, atmospheric haze, suggesting depth and hidden layers. Subtle rays of light or atmospheric effects hint at a hidden energy or underlying narrative. A subtle film grain and gentle halation add a touch of authenticity and nostalgia. Subjects, if present, are rendered in grayscale, with the exception of a single, striking red element.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Security Camera",
"cat": "Photo",
"pre": "You are a security camera operator.\nThe image is rendered in stark black and white with high contrast: bright whites and deep blacks, mimicking the limited dynamic range of low-resolution surveillance sensors.\nA faint pattern of horizontal scan lines is visible.\nThe lighting is flat and uniform, typical of fluorescent or ambient streetlights, with minimal shadows.\nThe image is taken from a high angle with fisheye distortion; the image is out of focus, and the main subjects appear small and distant, completely unsuspecting.\nA big timestamp appears in the upper right corner, using a simple monospaced numeric font. Overall, the image conveys an impersonal, clinical observation.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Modern Pin-Up photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer specializing in pin-up aesthetics.\nYour photo is modern, cutting-edge, but with a subtle style that evokes the sexy and glamorous look of pin-up.\nYour photo was taken in a studio, but it still captures a spontaneous moment that feels immediate and authentic. Subtle film grain and a soft halo add a touch of authenticity and nostalgia. Skin tones are realistic and warm, reflecting natural light.\nThe colors are vibrant yet warm (intense reds, greenish blues, pastel pinks, and buttery yellows). The background, on the other hand, is desaturated to keep the focus on the subject.\nThe composition has slightly off-center poses, daring angles, and suggestive postures.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Top-Tier Magazine Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a prominent fashion photographer and have worked for top-tier magazines. Your work is characterized by impeccable detail, striking lighting, and a sophisticated aesthetic. Your images showcase intricate textures in clothing, realistic and detailed skin, and captivating eye contact. The lighting combines hard and soft sources to create depth and drama. The colors are rich and vibrant, aiming to enhance the atmosphere and reflect the designer's vision. The composition is dynamic and visually impactful, often employing unconventional angles and framing. The overall feel is ambitious, glamorous, and undeniably chic.\nYOUR PHOTOGRAPH:\n<!--\n",
"post": "\n-->"
},
{
"name": "High-Key Fashion Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a fashion photographer who creates ultra-bright photos for luxury productions.\nYour photograph showcases a high-key composition shot flooded with soft studio lighting. The exposure is over-lit with white background, eliminating shadows and delivering a crisp, polished look. Colors are saturated yet controlled (vivid reds, cobalt blues, and crisp whites) while the photo has immaculate details thanks to high-resolution capture and minimal post-processing to maintain the airy, glamorous aesthetic.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Light and Airy Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer who loves bright, ethereal, and romantic images.\nYour photograph is bathed in soft, diffused light, creating a bright and airy aesthetic. Colors are pastel and delicate, with a focus on whites, creams, and light blues. The image features a shallow depth of field, blurring the background and drawing attention to the subject. Post-processing emphasizes highlights and reduces shadows, resulting in a clean, luminous look.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Nostalgic Warm Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a seasoned photographer creating images with a soft, vintage aesthetic reminiscent of 1970-80 film photography.\nThe photos display warm, slightly amber tones, gentle buttery highlights, and muted shadows, giving the scene a nostalgic, timeless feel.\nLighting is natural and diffused-golden-hour sunlight, overcast daylight, or window light filtered through sheer curtains-producing smooth, even illumination without harsh edges.\nExposure is balanced to preserve detail in both highlights and shadows, resulting in a moderate contrast curve that feels \"film-like\".\nA subtle haze or soft-focus effect adds a dreamy, painterly quality, while fine grain (similar to 400 ISO film) provides texture without distracting from the subject.\nOverall, the image looks intimate, elegant, and evocative, as if lifted from a cherished family album or a vintage lifestyle magazine.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Teal and Orange Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a color specialist photographer.\nYour photograph showcases a satured teal and orange colors. The image features cool teal tones in the shadows and highlights, balanced by warm orange tones in the midtones and skin tones, creating a vibrant photography with high contrast and vivid colors.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Ochre & Shadows Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer specializing in dramatic portraits with a vintage feel. Your signature style features strong contrasts and the interplay of light and shadow.\nYour photographs exhibit a single, strong light source illuminating one side of the subject's face, while the other side remains in deep shadow. The background is a warm ochre tone, dramatically overlaid with stark, high-contrast shadows resembling those cast by closed Venetian blinds. These shadows create a pattern of dark stripes across the background, adding a sense of mystery and confinement. The overall effect is nostalgic and evocative, emphasizing the subject's expression and inner world.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Orthochromatic Spirit Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer inspired by the look of early photographic processes, specifically orthochromatic and colorized films.\nYour photograph showcases a scene with a high-contrast, blue-sensitive aesthetic. Reds appear very dark, while blues and greens are bright. The image is characterized by a stark, graphic quality. Post-processing focuses on enhancing the contrast and emphasizing the tonal separation.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Synthwave Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer inspired by the retro-futuristic visuals of the 1980s.\nYour photograph showcases Synthwave aesthetic, characterized by neon colors and geometric shapes, it includes vibrant pinks, purples, and blues, with a glowing effect. Grids, scanlines, and a retro-futuristic atmosphere. The photograph feels like a scene from a classic 80s sci-fi film.\nReal people against a computer-generated background.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Quiet Luxury Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are an editorial fashion photographer specializing in the Old Money aesthetic. \nYour photograph showcases a sophisticated, clean composition using natural, soft morning light. The color palette is muted and neutral, focusing on beige, cream, navy blue, and forest green. The scene feels expensive and timeless, set in a luxury location. \nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Daylight Paparazzi",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photojournalist specializing in candid street photography, often working from a distance with a telephoto lens.\nYour photographs are characterized by a compressed perspective, a shallow depth of field, and a sense of voyeurism. The subject is often unaware of being photographed, caught in a natural moment.\nThe composition frequently includes foreground elements - trees, cars, people - that partially obscure the view, creating a sense of being hidden or observing from afar. Expect a focus on capturing authentic expressions and interactions, even if slightly blurry due to the distance. The overall feeling is one of observation and discreet documentation.\nYOUR PHOTO:\n<!--\n",
"post": "\n-->"
},
{
"name": "Nightlife Paparazzi",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a paparazzi photographer specializing in nighttime shots of celebrities.\nYour photographs are characterized by the harsh, direct flash, combined with a grainy, high-ISO aesthetic. The background is often blurred and indistinct, emphasizing the subject.\nThe composition is often chaotic and dynamic, reflecting the frenetic energy of the scene. Expect red-eye, blown-out highlights, and a sense of intrusion.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Dark-Side Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer who creates images that exude tension and mystery.\nThe scene is lit by a single, strong side light that cuts sharply across the subject, producing deep, inky shadows on the opposite side and a dramatic rim of light that outlines form.\nThe background is completely dark, absorbing any spill and leaving only faint silhouettes or subtle gradients that recede into blackness.\nHighlights are crisp and slightly over-exposed, giving a cold, almost clinical glow, while the shadow side retains rich texture and detail without flattening.\nA thin veil of low-level atmospheric haze may drift near the lit edge, adding depth without softening the harsh contrast.\nColors are desaturated except for a narrow band of cool-blue or muted-green tones that may appear in the illuminated areas, heightening the unsettling mood.\nA subtle, fine grain reminiscent of high-ISO film adds a gritty texture that reinforces the feeling of unease.\nOverall, the photograph feels like a still from a thriller; that is, intense, claustrophobic, and loaded with suspense.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Dramatic Light & Shadow",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a professional photographer specializing in high-impact visual storytelling, with a focus on dramatic lighting and strong contrast.\nYour images feature ultra-sharp detail and a color palette that emphasizes deep blacks, punchy primary colors, and selective color isolation.\nThe lighting is meticulously sculpted: a dominant hard rim-light or a strong directional key-light creates pronounced shadows, while a subtle fill-light preserves texture. \nYou deliberately push the exposure to achieve a slight over-exposure in the highlights, producing a glowing effect that draws the eye.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Pastel Dream Aesthetic",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a contemporary photographer whose work is crafted for a modern, social-media-savvy audience.\nThe images are clean, sharply focused, and highly polished, with a crisp resolution that highlights fine detail.\nLighting is bright and evenly diffused, often using natural daylight or soft artificial key-light that produces a luminous, airy feel.\nColors are vivid yet harmonious: pastel-toned highlights (peach, mint, blush) sit alongside saturated mids tones (turquoise, coral, mustard) that pop without looking oversaturated.\nA subtle, soft vignette gently darkens the edges, drawing the eye toward the centre and emphasizing the main subject.\nContrast is moderate-high, giving the picture depth while preserving details in both shadows and highlights.\nA faint, almost imperceptible grain (similar to fine 200 ISO film) adds a touch of texture, keeping the image from feeling too clinical.\nOccasional light-leak flares or bokeh circles appear in the background, providing a playful, spontaneous element.\nComposition follows the \"rule of thirds\" or centered framing, with plenty of negative space to create a balanced, Instagram-ready aesthetic that feels both aspirational and approachable.\nOverall, the photograph feels fresh, stylish, and emotionally uplifting, ready to capture attention at a glance.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Theatrical Illusion",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a set designer creating a theatrical scene to visual impact.\nYour photographs exhibit an aesthetically pleasing artificial environment. The lighting is dramatic and carefully orchestrated, with strong contrasts and a focus on creating mood.  Elements are deliberately exaggerated. There's a sense of constructed beauty, as if the scene exists solely for the viewer's gaze. Subtle effects like mist or smoke enhance the dreamlike quality. The overall impression is one of a meticulously crafted illusion, a stage set brought to life.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Exposed Film Set",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer documenting a film set in the midst of production. Your work embraces a deliberately unfinished and revealing aesthetic. Painted backgrounds are visible, with intentionally rough or incomplete areas. Lighting rigs and studio equipment are partially in frame. Props are clearly artificial, revealing their construction and imperfections. Along the right edge, the set boundary opens up to reveal the film crew and equipment against a black backdrop. The scene feels exposed, as if the audience has wandered onto the set during a take. The overall effect is a playful deconstruction of the filmmaking illusion, revealing the process behind the magic.\nYOUR PHOTO:\n<!--\n",
"post": "\n-->"
},
{
"name": "Burning Man Playa",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nA cinematic photograph that captures the ethereal beauty and radical expression of Burning Man. The scene is bathed in golden hour light, focusing on the vastness of the playa and the stark contrast between art, technology, and the desert landscape. Subtle dust particles float in the air. The overall image is sharp and detailed, with a focus on capturing the textures of the desert and the art. The aesthetic is modern, clean, and emphasizes a sense of scale and awe.\nTHE PLAYA:\nThe playa is an expanse of blindingly white alkaline dust. The horizon is sharp and clear, stretching to infinity. A distant, slightly blurred line of mountains is visible. A weathered vintage pickup truck sits parked in the distance, partially obscured by the haze. Other distant vehicles are subtly discernible, suggesting a bustling community.\nTHE LIGHTING:\nThe warm light of golden hour dominates the scene, casting long, dramatic shadows. Subtle gradients of color - peach, rose, and lavender - paint the sky.  Flashes of neon light emanate from the art installations, creating bursts of vibrant color.\nTHE ART AND THE PARTICIPANTS:\nImpressive art installations accentuate the landscape, showcasing innovative design and interactive elements. A crowd of participants stand out as silhouettes in unique, futuristic attire. The focus is on the interaction between art and the environment with hundreds of people on the horizon.\nTHE PHOTO:\n<!--\n",
"post": "\n-->\nTHE FEELING:\nA sense of wonder and freedom. The photo evokes the feeling of being lost in a dream, surrounded by creativity and innovation."
},
{
"name": "Street Documentary Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a documentary photographer chronicling urban life.\nYour photograph showcases candid moments in natural street environments. The image employs a natural color balance, modest contrast, and a shallow depth of field to isolate subjects within bustling scenes. Emphasis is placed on authentic emotion, body language, and the interplay of ambient city light, such as neon signs and street lamps. Minimal post-processing preserves the raw, honest feel, only slight exposure adjustments and noise reduction are applied to maintain the immediacy of the moment.\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Tilt Shift / Toy Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a photographer who enjoys using tilt-shift lenses to turn real-world scenes into miniature toy models.\nYour photographs exhibit crisp, high-resolution details captured in an exaggerated perspective, with a very narrow focus plane, and a dreamy, toy-like atmosphere that creates the classic \"miniature\" illusion.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Pop Photo",
"cat": "Photo",
"pre": "YOUR CONTEXT:\nYou are a contemporary pop artist known for your vibrant, oversaturated palettes and energetic, graphic compositions.\nYour photographs exhibit exaggerated colors and solid backgrounds using Ben-Day dots, achieving an unconventional and clearly pop art effect.\nYOUR PHOTO:\n",
"post": ""
},
{
"name": "Comic 1",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an American artist.\nYour illustration has a comic book cover style, with vibrant colors and outlined shapes, featuring realistic and epic details in the style of modern comics.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Comic 2",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an Italian artist.\nYour illustration has a European comic book style, featuring black hand-drawn outlines, solid colors, realistic anatomical proportions, and cross-hatching for shading.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Comic 3",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a European artist creating fiction worlds.\nYour image is a classic ligne claire futurist illustration, featuring exceptionally clean lines that define each element with the utmost precision. The composition highlights an intense atmosphere with complex technology. The color palette is vibrant, with flat colors and intricate lines and design.\nYOUR IMAGE:\n",
"post": ""
},
{
"name": "Action Cover Art",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a cutting-edge comic artist specializing in high-octane, mature industrial sci-fi where movement is everything.\nYour illustration captures a moment of extreme tension featuring \"ink splatters\" that suggest violent, rapid motion. The style is technical and sharp, with a focus on high-tech gear, tactical clothing, and urban decay, all rendered with a jittery, energetic hand-drawn line. The composition uses extreme low-angle shots to make the subjects feel powerful and intimidating, while the lighting mimics the cold, flickering strobes of an industrial complex, adding a sense of claustrophobic urgency.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Explosive Cover Art",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a celebrated comic book cover artist, renowned for hyper-realistic action and monumental heroics.\nYour cover art is a Dynamic Heroic Realism illustration, characterized by intricate anatomical accuracy, meticulously rendered textures (fabric, metal, skin), and an explosive sense of motion. The composition often features dramatic foreshortening and a heroic, powerful pose for the central figure, surrounded by dynamic environmental destruction or energy effects. Lighting is grand and directional, sculpting forms with sharp highlights and deep shadows, ensuring every detail contributes to an overwhelming sense of epic scale and immediate impact.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Intense Pop Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nA vibrant, slightly imperfect illustration reproduced using a stencil printing process, resulting in vivid, flat colors, visible registration errors, and a unique texture. It also features a limited color palette and a slightly grainy appearance.\nLAYERS AND COLORS:\nThe image is built from layers of color with a limited palette of reds, blues, yellows, and blacks. The colors overlap and blend unexpectedly due to misregistration.\nTHE ILLUSTRATION:\n",
"post": "\nTEXTURE AND IMPERFECTIONS:\nA grainy texture with small dots, smudges, and variations in ink density contributes to the handcrafted feel.\nTHE OVERALL FEEL:\nA raw, energetic, and slightly retro aesthetic. The image feels handmade, expressive, and full of character. Think fanzines, concert posters, and independent art prints."
},
{
"name": "Pop-Art",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a bold artist.\nYour illustration features Pop Art style, characterized by bold vibrant colors. The colors are executed using the pointillist technique, where dots of color come together to create the overall picture. Dazzling patterns and comic style characters. The scene has a flat, two-dimensional appearance typical of pop art aesthetics.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Vintage Comic",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a 1950s artist.\nYour illustration in a worn sheet of paper features a vintage 1950s comic book style made with hand-drawn lines and painted in solid white, red, blue, and black.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Vintage Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a sci-fi pulp magazine illustrator from the 1940s.\nYour drawing is a lush, painted-style comic cover including dramatic, heroic lighting and a vivid \"Technicolor\" palette of saturated oranges, deep teals, and bright yellows. It features soft airbrushed textures, muscular anatomy, and a grainy film-stock finish that evokes a sense of retro-futuristic adventure.\nYOUR DRAWING:\n",
"post": ""
},
{
"name": "Modern Pin-Up Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a specialist in retro-inspired erotic illustration.\nYour artwork follows a Vintage Pin-up style, featuring a charismatic figure posed provocatively within fields of stylized geometry and chromatic gradients. The composition is balanced and symmetrical, employing the \"rule of thirds\" to place the figure's gaze and gesture in a way that invites the viewer's eye across the frame. Lines are sleek and fluid, emphasizing the form and musculature of the figure. Lighting is soft yet directional, reminiscent of studio photography: a luminous aura envelops the skin, while subtle atmospheric shading softens contours, yielding a glossy, almost photographic finish. Color choices are vibrant yet tasteful, mirroring the palette of classic pin-up posters. Background elements, such as checkerboard patterns, tonal washes, abstracted forms, are rendered with minimal detail, using gentle gradients and subtle texture to maintain focus on the figure's dynamic pose.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Manga",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a Japanese manga artist.\nYour illustration features black and white manga-style that includes manga screentone patterns with high-contrast black and white drawing, in a violent atmosphere. Always imagine colors as shades of gray.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Anime",
"cat": "Illustration",
"pre": "The illustration has the following characteristics:\nDynamic Composition: Iconic, atmospheric layouts emphasizing scale & world-building. Uses verticality/triangular hierarchies, focusing on impactful stillness & dramatic perspective, e.g: extreme high/low angles.\nPrecise Inking: Sharp, digitally precise line art with variable weights for depth. Thick outlines define forms, ultra-fine lines detail interiors, & lines subtly shift/blend for light effects.\nStylized Anatomy: Proportions adhere to series character designs (shōnen/seinen). Prioritizes instantly recognizable silhouettes with graceful, defined forms emphasizing agility.\nHigh Detail Rendering: Technical detail exceeding animation standards. Flawless character skin contrasts with intricately rendered hardware/environments - layered hair, jewel-like eyes with complex reflections.\nAdvanced Shading: Multi-tone shading with ambient occlusion & rim lighting. Volumetric light creating depth in a 2D space, e.g: sunbeams, flares, particles. \nNarrative Layout: Thematic, vertical layouts with balanced composition. Uses negative space to guide the eye & character positioning to convey relationships/story.\nSoulful Expression: Introspective acting focused on subtle emotion in eyes & hands. Avoids exaggeration, favoring still compositions conveying themes like longing or resolve.\nVibrant Color & Texture: Detailed skin texture with intricate shadows. Enhanced with cinematic effects, e.g: aberration, grain, bloom, motion blur.\nTHE ILUSTRATION:\n<!--\n",
"post": "\n-->"
},
{
"name": "Dark Anime",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a Japanese anime artist.\nYour Japanese animation includes an ominous atmosphere, clean lines, and dramatic facial expressions. The artwork uses cool and dark tones along with dramatic lighting and shadows that emphasize the central figure and create a somber, intense mood.\nYOUR DRAWING:\n",
"post": ""
},
{
"name": "Ultimate Anime",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a top-tier anime key visual artist specializing in high-fidelity CGI rendering. Your goal is to create promotional artwork that blends the expressive character designs of anime with the polished look of modern 3D animation.\nThe artwork is rendered at 8K (or higher) resolution with perfectly clean linework and seamless anti-aliasing. Color palettes are vibrant and harmonious, adhering to established character designs and world aesthetics.\nLighting is *critical*: a strong key light defines form, rim lighting separates the subject, and soft fill lights preserve detail. Highlights should have a pronounced bloom and subtle lens flares. Shadows are smoothly graded with a focus on realistic light diffusion.\nShading utilizes advanced rendering techniques like physically-based rendering (PBR) to create realistic material properties. Surfaces (skin, hair, fabric, metal) have a polished, three-dimensional feel with subtle subsurface scattering and detailed specular highlights. Avoid overly-stylized cell-shading; aim for a smooth gradient.\nBackgrounds are richly composed, utilizing depth-of-field, atmospheric perspective, and subtle particle effects (digital glitches, energy particles, volumetric lighting) to enhance immersion. Backgrounds should *support* the characters, not distract from them.\nLinework is clean and confident, with varying line weights to emphasize form. Consider a subtle digital \"smoothing\" effect to the lines.\nComposition is dynamic and emotionally resonant: strong poses, expressive facial features like bright expressive anime eyes, and soft gradients and dynamic strands.  Think about the \"rule of thirds\" and leading lines.\nThe overall impression should be that of a high-budget anime promotional image rendered with cutting-edge CGI technology - visually striking, technically flawless, and deeply respectful of the source material. Consider elements like bloom, lens flares, and subtle chromatic aberration.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Studio Anime",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a Japanese anime artist.\nYour Japanese animation includes warm artwork inspired by Studio Ghibli, with detailed characters and a whimsical aesthetic, full of emotion.\nYOUR DRAWING:\n",
"post": ""
},
{
"name": "Retro Anime 80s",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a nostalgic animator who revives the iconic visual language of 1980s Japanese anime. Your artwork captures the era signature aesthetic: bold, angular designs with exaggerated proportions, vibrant primary colors (crimson red, cobalt blue, sunshine yellow), and dramatic, hand-drawn line hatching. The backgrounds feature expansive vistas of geometric objects, and stylized atmospheric formations rendered in a grainy, slightly desaturated texture reminiscent of cel-animation cels. Lighting is dramatic, with strong edge highlights that define form and spark-filled energy emissions, while chromatic aberrations punctuate areas of intense power.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Simple Toon",
"cat": "Illustration",
"pre": "The illustration has the following characteristics:\nBold, thick outlines define shapes with a strong cartoon aesthetic.  Stylized anatomy with exaggerated proportions (large heads, rounded bodies). Minimal detail, focusing on essential outlines and simple props.  Shading is achieved with uniform diagonal lines.  Expressive characters with exaggerated facial features (big eyes, lively eyebrows) and concise body language.  Flat, saturated colors with smooth surfaces.  Emphasis on dynamic composition and clear visual flow.\nTHE ILUSTRATION:\n<!--\n",
"post": "\n-->"
},
{
"name": "Joyful Cartoon Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a painter creating a humorous artwork inspired by the vibrant and expressive style of Pixar animation. Your paintings are full of exaggerated features, dynamic poses, and playful details.\nYour paintings exhibit bright, saturated colors, soft lighting, and a sense of depth and dimension. Characters are expressive and emotive, with exaggerated facial features and body language. The composition is dynamic and engaging, with a focus on storytelling and visual humor. The overall aesthetic is cheerful, whimsical, and full of energy.\nYOUR PAINTING:\n",
"post": ""
},
{
"name": "Dark Comic Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a master of hard-boiled noir graphic novels, specializing in themes of vengeance and the urban underworld. \nYour illustration is a masterclass in high-contrast chiaroscuro, featuring heavy, ink-drenched shadows and sharp, brutal highlights. The style is raw and visceral, with a focus on shiny surfaces, textured grime, and intense facial expressions that convey deep psychological weight. The composition is cinematic and claustrophobic, using low-angle shots to empower characters and stark lighting to create a sense of impending danger and adult narrative depth.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Digital Cyberpunk",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a digital artist working in a dystopian, technologically advanced future.\nYour illustration is a Cyberpunk Noir scene, characterized by harsh, dramatic lighting that casts long, sharp shadows, reflecting the grimy underbelly of a futuristic scene. It employs a limited color palette dominated by deep blues, neon greens, and stark oranges, giving a mysterious aesthetic.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Electric Blue Outline",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYour image showcases a digital cyberpunk draw, characterized by vibrant and saturated colors with solid backgrounds and blue outlines.\nYOUR IMAGE:\n",
"post": ""
},
{
"name": "Neon Rampage Illustration",
"cat": "Illustration",
"pre": "Your illustration has the following characteristics:\n- Explosive composition with wild angles.\n- Dynamic lines with irregular, overlapping strokes of varying thickness.\n- Distorted anatomy with absurd proportions, e.g: elongated limbs, impossible contortions.\n- Neon lighting with saturated colored light sources and shadows in blocks of solid color.\n- Extreme expressiveness with exaggerated facial expressions.\n- Vibrant palette that includes a combination of primary colors with fluorescent tones and black/white contrasts.\nYOUR ILLUSTRATION:\n<!--\n",
"post": "\n-->"
},
{
"name": "Ink Frenzy",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an underground street-art comic illustrator.\nYour image is a high-energy, chaotic ink drawing featuring aggressive brush strokes, intentional ink splatters, and \"controlled mess\" textures. It combines thick, variable-width outlines with vibrant, clashing spray-paint colors. The style is rebellious, modern, and feels like it's vibrating with motion and urban grit.\nYOUR IMAGE:\n",
"post": ""
},
{
"name": "Ink & Shadow",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a master of hard-boiled noir illustration, prioritizing raw emotional impact through minimalist techniques.\nYour illustration is a brutalist, hand-inked work featuring stark black-and-white contrasts with a single, aggressive accent color (like blood red or electric gold). Lines are thick, confident, and jagged, stripping away detail to focus on character and environment. The composition is cinematic and voyeuristic, emphasizing shadows and reflective surfaces.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Unsettling",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an illustrator of dark and disturbing themes.\nYour illustration includes stippling textures, dramatic raking side lighting, and a dread-filled atmosphere that is oppressive and claustrophobic. High contrast emphasizes form and shadow, employing cinematic chiaroscuro for depth.\nYOUR DRAWING:\n",
"post": ""
},
{
"name": "Digital Engraving",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a graphic designer and printmaker, celebrated for crafting traditional engravings.\nILLUSTRATION BACKGROUND:\nA minimal, iconographic background. Simple shapes and flat colors, suggesting a location or environment without distracting from the main subject.\nYOUR ILLUSTRATION:\nYour illustration features a striking monochrome art style, characterized by bold, crisp black outlines and intricate stippling or halftone dot patterns that provide nuanced shading and texture. The aesthetic evokes the meticulous detail of a linocut or woodcut, rendered in high-contrast black and white. Faces, if present, are rendered with a degree of realistic detail, balancing the graphic style with recognizable human features. The composition is impactful, designed for immediate recognition and a timeless, graphic appeal. Significant areas of white space are incorporated to emphasize the design and create a sense of elegance and clarity. The foreground is highly detailed and textured.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Digital Risograph",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an underground indie press printer.\nYour illustration is a tactile Risograph-style print, featuring vibrant fluorescent ink and a heavy grainy texture. It uses a punchy palette of neon pink, intense yellow, and deep teal.\nYOUR MULTI SUBJECT PRINT:\n",
"post": ""
},
{
"name": "Surreal Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nAn illustration plucked straight from the illogical and surreal landscape of a dream. The scene defies gravity and conventional perspective. Objects transform and merge into one another. The colors are vibrant and unexpected.\nThe scene appears distorted and fragmented.\nTHE SCENE:\n<!--\n",
"post": "\n-->\nSURREALIST ELEMENTS:\nFloating islands, melting clocks, upside-down buildings, animals with human features, impossible geometry.\nBACKGROUND:\nA whirlwind of colors and shapes suggesting a boundless and chaotic space.\nSTYLE:\nInspired by Salvador Dalí and René Magritte, with a touch of children's illustration."
},
{
"name": "Neo Cubismo",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an American artist.\nYour work has a Cubist style, with geometric and abstract forms.\nYou emphasize fragmented, multi-view facial geometry: you divide each face and body into angular fragments, and distorted shapes.\nYou avoid soft skin tones, photorealistic shading, or realistic contours; instead, you use flattened blocks of color, muted tonal contrast, and exaggerated angular distortion.\nYou eliminate any trace of photographic texture; you replace it with matte, painterly surfaces and bold contour strokes that separate each facet.\nYou ireassemble face features (eyes, nose, mouth) as disjointed, overlapping forms. You use a restricted, non-naturalistic palette (e.g., cadmium red, ultramarine, ochre, black) applied with flat, even washes.\nIn general, the composition should have an abstract, psychedelic and deformed appearance.\nYOUR ABSTRACT ARTWORK:\n",
"post": ""
},
{
"name": "Epic Greg",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a master digital painter specializing in grand-scale heroic fantasy illustrations. Your work evokes the spirit of classic fantasy artists like Frank Frazetta, Boris Vallejo, and Brom.\nYour illustration is a meticulously crafted digital oil painting, characterized by masterful, sweeping brushwork and thick impasto textures that give a sense of tangible depth. It features dramatic, cinematic lighting with intense chiaroscuro: light sources like roaring fires, magical energy, or celestial bodies pierce through atmospheric haze, billowing smoke, and swirling clouds.\nThe composition is classically inspired, often employing dynamic poses, heroic proportions, and a sense of epic scale.  Details are richly rendered, focusing on textures like weathered armor, flowing fabrics, and rugged landscapes. The palette is a moody blend of deep earth tones (ochres, umbers, siennas) accented by vibrant, glowing highlights (gold, crimson, azure) to create a legendary and immersive feel.\nConsider incorporating elements of high fantasy tropes: mythical creatures, ancient ruins, powerful artifacts, and characters embodying courage, strength, and destiny.\nYOUR PAINTING:\n",
"post": ""
},
{
"name": "Cyber Idol",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an avant-garde Japanese concept artist specializing in futuristic idol culture. Your illustrations fuse the dazzling charisma of J-pop idols with a tech-saturated aesthetic, rendered in a dynamic digital painting style with visible brushstrokes. The visual scheme features a kaleidoscope of neon pinks, electric cyan, ultraviolet, and hot magenta, contrasted against deep, metallic blacks and chrome surfaces, all with a painterly quality. Characters sport elaborate cybernetic accessories, like holographic projections and light-emitting garments, rendered with stylized reflective materials and exaggerated glowing emissive effects. The lighting is multi-layered: radiant atmospheric glows, dynamic energy flares, and shifting chromatic beams that dissect the haze, interpreted with bold color and dramatic contrast. Compositionally, you combine tight, charismatic portrait shots with wider views, using *suggested* motion blur and particle effects to convey a pulsating, hyper-connected performance. The overall aesthetic leans towards a vibrant, energetic illustration reminiscent of contemporary anime keyframes.  The background is abstract and dynamic, focusing on light and color rather than a specific location.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Minimalist Digital Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a cover artist known for powerful, minimalist designs.\nYour cover art is a bold graphic iconography piece, featuring stark, impactful silhouettes and clean, sharp lines that define forms with absolute clarity. The color palette is vibrant, employing large fields of solid color with subtle gradients. Most importantly, the proportions are exaggerated, and the perspective is extremely distorted, creating a sense of vertigo and action.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Synthwave Digital Illustration",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an eclectic digital artist inspired by 80s nostalgia and pastel futurism.\nYour illustration follows a Retro Vaporwave aesthetic, featuring pastel gradients, glitchy raster effects, and abstracted retro signifiers such as fragmented landscapes, digital artifacts, and chromatic distortions. The composition is flat yet layered, with glowing grid lines and reflective surfaces that evoke a dreamy, nostalgic ambience.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Whimsical Watercolor",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are an illustrator specializing in gentle, enchanting narratives for children's books.\nYour drawing is a whimsical watercolor image, featuring loose, fluid watercolor washes, delicate linework, and playful, anthropomorphic elements within a lush natural setting. The style is soft and inviting, with a vibrant yet gentle color palette that captures the magic of innocence.\nYOUR DRAWING:\n",
"post": ""
},
{
"name": "Modern Ukiyo-e Print",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a master of traditional Japanese woodblock printing adapted to a modern narrative.\nYour illustration features bold, flowing outlines and flat areas of color with subtle wood-grain textures. The composition uses dramatic perspective typical of \"Ukiyo-e\" art, with stylized waves, clouds, or smoke, and a color palette of deep indigo, vermillion, and ochre on an aged washi paper background.\nYOUR DRAWING:\n",
"post": ""
},
{
"name": "LowRes Pixel Art",
"cat": "Illustration",
"pre": "YOUR CONTEXT:\nYou are a pixel artist.\nYour illustration is a low-resolution pixel art featuring chunky pixels and vibrant intense colors, arcade game style.\nYOUR ILLUSTRATION:\n",
"post": ""
},
{
"name": "Fridge Magnet",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA brightly colored, plastic refrigerator magnet. The magnet is slightly curved to adhere to the metallic surface of a refrigerator. The plastic has a slight sheen, and the edges are rounded. The background is a stainless steel refrigerator door, with subtle reflections of the kitchen.\nTHE DRAWING ON THE MAGNET:\nA flat, vector-style illustration. Bold outlines and simple color fills. The illustration is playful and cheerful.\nTHE ILLUSTRATION:\n",
"post": "\nTHE REFRIGERATOR BACKGROUND:\nA stainless steel refrigerator door with subtle reflections of a kitchen environment."
},
{
"name": "Low-Poly Render",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nYou create a Unity Engine render of a low-poly 3D retro game, built with low-poly meshes and low-quality textures.\nINSIDE YOUR RENDER:\nAll elements in the scene are low-poly; some are even simple cubes. All characters in the scene are low-poly objects.\nINSIDE YOUR RENDER:\n<!--\n",
"post": "\n-->"
},
{
"name": "Simple 3D Render",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nYou are a 3d artist.\nYour render is a high-gloss, illustration featuring ultra-reflective surfaces. The style is dark with dark backgrounds and iridescent highlights.\nYOUR RENDER:\n",
"post": ""
},
{
"name": "Vintage VGA Monitor",
"cat": "Wild",
"pre": "YOUR CONTEXT\nAn old VGA CRT rectangular monitor sitting on a wooden desk, and a window is reflected on its screen. The monitor's casing has a raised black inscription in the upper left corner that reads \"VGA\". The screen has a slight curve and sweep lines typical of CRT displays.\nINSIDE YOUR MONITOR'S SCREEN:\nA low-resolution, 2D pixel art platform videogame form early 1990s.\nINSIDE THE VIDEOGAME FROM EARLY 1990s:\n",
"post": "\nOUTSIDE YOUR MONITOR'S SCREEN:\nA wooden desk and a wall."
},
{
"name": "Vintage Polaroid Photo",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nYou are a nostalgic photographer who embraces instant film's tactile charm.\nYour Polaroid photograph is using vintage-style film. The photograph features the iconic square format including a soft focus, a subtle vignetting, and the characteristic pastel color shift of instant film. Showing candid moments or intimate still-lifes. The photograph retains the authentic border, emulsion texture, and occasional \"film-like\" imperfections (light leaks, scratches).\nYOUR PHOTOGRAPH:\n",
"post": ""
},
{
"name": "Ancient D&D Character Sheet",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nAn aged, parchment-style Dungeons & Dragons character sheet. The paper is heavily textured, with visible creases, stains, and a slightly yellowed hue. The edges are frayed and uneven. The overall aesthetic is one of ancient lore and forgotten adventures. The sheet rests on a black piece of wood.\nLAYOUT:\nThe sheet is divided into two main sections. The left side contains character statistics and information, while the right side features a full-body illustration of the character. The background is a complex pattern of faded ink and subtle textures.\nLEFT SIDE - CHARACTER INFORMATION:\nAt the top, a large, ornate title area for the character's name, then a separator line, and below that a descriptive paragraph in brown font for: [CHARACTER DESCRIPTION].\nBelow the description, a rectangle with a table showing the character's statistics:\n - STR:\n - DEX:\n - CON:\n - INT:\n - HP:\n end\nThe text is written in a calligraphic font, with flourishes and embellishments.\nRIGHT SIDE - CHARACTER ILLUSTRATION:\nA dramatic, full-body illustration of the character, rendered in a highly detailed and dynamic style. The character is posed in a heroic or imposing manner, with a sense of power and presence. The illustration is surrounded by a swirling, ethereal background, suggesting magical energy or a dramatic environment.\nTHE ILLUSTRATION:\n",
"post": "\nBACKGROUND DETAILS:\nFaded ink patterns, arcane symbols, subtle textures resembling ancient maps or scrolls. The overall background should complement the character illustration and enhance the sense of fantasy and adventure. The paper have black corner flourishes."
},
{
"name": "Retro Arcade Cabinet",
"cat": "Wild",
"pre": "YOUR CONTEXT\nA close-up of an early 1990s arcade machine. It sits in a dimly lit arcade, its side panels covered in glossy vinyl graphics. The CRT monitor glows with its characteristic phosphorescent hue, and the joystick and buttons show slight wear. The arcade machine is viewed from a three-quarter angle, capturing its design, the bright screen, and the reflective floor tiles.\nINSIDE YOUR CRT MONITOR:\nAn action-packed, pixelated, yet brightly colored arcade game.\nINSIDE THE PIXELATED ARCADE GAME:\n<!--\n",
"post": "\n-->\nOUTSIDE THE PIXELATED ARCADE GAME:\nA dimly lit arcade, including other machines and neon signs, both slightly out of focus.\nTHE PHOTO:\nA close-up of an early 1990s arcade machine. It sits in a dimly lit arcade, its side panels covered in glossy vinyl graphics. The CRT monitor glows with its characteristic phosphorescent hue, and the joystick and buttons show slight wear. The arcade machine is viewed from a three-quarter angle, capturing its design, the bright screen, and the reflective floor tiles."
},
{
"name": "Retro Pinball Backglass",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA close-up, slightly tilted view of a classic 1980s pinball machine. The focus is on the illuminated back panel. The machine is in a dimly lit arcade, with other machines and diffused neon lights in the background. The cabinet is worn, showing signs of use. At the bottom of the image, the machine is visible with shiny metal rails and a slightly dirty play surface. At the top, the back panel is slightly scratched and faded, giving it a retro feel.\nINSIDE YOUR BACK PANEL ART:\nA vibrant and highly detailed illustration in the style of 1980s arcade art. Vibrant colors, dynamic poses, and a sense of action. The art features a powerful light source from behind, creating a bright effect. The style is reminiscent of artists like Boris Vallejo or Frank Frazetta, but adapted to the pinball theme. Embedded in the illustration are red LED panels displaying the score 10320423.\nINSIDE THE VIBRANT ILLUSTRATION:\n<!--\n",
"post": "\n-->\nOUTSIDE THE VIBRANT ILLUSTRATION:\nThe arcade background is out of focus, with hints of other machines and neon lights.\nTHE PHOTO:\nA close-up, slightly tilted, of a classic 1980s pinball machine. The focus is on the illuminated back panel. The machine is in a dimly lit arcade, with other machines and neon lights out of focus in the background. The cabinet is worn, showing signs of use and age."
},
{
"name": "Urban Wall Mural",
"cat": "Wild",
"pre": "SCENE:\nA huge mural on a rough brick wall. The scene is captured from a slightly low angle, highlighting the brick texture and spray paint drips. Ambient city light (neon signs, streetlights) blends with natural light, giving the wall a vibrant, kinetic feel.\nINSIDE THE MURAL:\nAn intense, modern spray-painted illustration.\nINSIDE THE ILLUSTRATION:\n",
"post": "\nOUTSIDE THE MURAL: The brick wall, a window high up, paint cans on the ground, and a weathered patina. The surrounding alley and someone admiring the artwork are visible to the side."
},
{
"name": "City Light Poster",
"cat": "Wild",
"pre": "YOUR PHOTO:\nAn eye-level, realistic photo of a glass-encased advertising poster (mupi) located on a clean city sidewalk. There are reflections of nearby buildings and trees on the glass surface. The lighting is natural overcast daylight. The advertising poster is faded and sun-faded, low quality printed advertisement.\nPOSTER DETAILS:\nWhile the street and buildings are visible in the background, the inside of the poster contains the following:\n",
"post": ""
},
{
"name": "Neon Billboard",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA massive neon-lit billboard dominates a bustling downtown avenue at twilight. The billboard's glass casing reflects nearby traffic and skyscraper windows. The atmosphere is saturated with city haze, rain-slicked streets, and colorful streetlights.\nDISPLAYED ON THE BILLBOARD (YOUR ADVERTISEMENT):\n",
"post": "\nYOUR PHOTO:\nAn eye-level, cinematic photograph of the billboard from street level, showing the vivid neon glow, the wet pavement reflections, and distant cityscape. The billboard's content is crisp and bright, while the surrounding environment retains a slight motion blur for a dynamic urban feel."
},
{
"name": "Jigsaw Puzzle",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA nearly completed jigsaw puzzle lying on a warm-toned wooden table. The puzzle pieces are made of glossy cardboard, reflecting the light from a nearby window. One puzzle piece is missing, and it rests beside the puzzle on the table. The scene is lit with soft, natural light.\nTHE PUZZLE IMAGE:\nA detailed, high-resolution image of ",
"post": ". The image is vibrant and colorful, with sharp details.\nTHE MISSING PIECE:\nThe missing puzzle piece is lying on the table, slightly angled, showing its unprinted back.\nTHE TABLE & REFLECTIONS:\nA wooden table with a visible grain. The glossy surface of the puzzle reflects the light from a window, creating highlights and subtle distortions."
},
{
"name": "Christmas Window Display",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA beautifully decorated Christmas window display in a traditional toy store, situated on a street corner. The scene is viewed from across the street at night, with a slight blur from the distance and the cold winter air. Snow is gently falling. The window is large and showcases a detailed miniature world.\nTHE DISPLAY INSIDE THE WINDOW:\nA festive scene with intricately crafted decorations, including miniature buildings, snow-covered trees, and twinkling lights. The overall aesthetic is warm and inviting.\nTHE CENTRAL FIGURE:\n",
"post": " is the focal point of the display, surrounded by charming Christmas decorations.\nTHE SURROUNDING DECORATIONS:\nSnow-covered miniature buildings, twinkling fairy lights, frosted pine trees, and delicate ornaments.\nTHE EXTERIOR:\nThe scene is a street corner. To the right, a blurred street with softly lit storefronts. Above and below the window frame, dark architectural elements (like ledges or cornices) are visible, retaining a layer of freshly fallen snow. The overall exterior is a cold, wintery night scene with a subtle glow from streetlights."
},
{
"name": "Classic Snow Globe",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA beautifully crafted glass snow globe sitting on a polished black wooden table. The globe is filled with swirling, glittering snow. The base of the globe is dark wood with a small, engraved silver plaque. In the blurred background, a softly lit Christmas tree is visible. The lighting is warm and inviting.\nINSIDE THE SNOW GLOBE:\nA miniature, highly detailed scene featuring a charming character. The scene is slightly vintage in style, evoking a sense of nostalgia.\nTHE CHARACTER AND SCENE:\n",
"post": "\nTHE GLOBE'S BASE PLAQUE:\nEngraved with a delicate, cursive font.\nTHE BACKGROUND:\nA softly blurred Christmas tree with twinkling lights, creating a festive atmosphere."
},
{
"name": "90s Cereal Box",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA brightly colored cereal box, reminiscent of the 1990s. The box is made of cardboard with a glossy finish. It's sitting on a Formica kitchen countertop, slightly angled towards the viewer. There's a hint of a checkered tablecloth in the background. The box features bold typography and a cartoon mascot.\nTHE CEREAL BOX DESIGN:\nThe box is filled with vibrant colors and dynamic illustrations. The cereal name is prominently displayed in a playful font. A cartoon mascot interacts with the cereal pieces.\nINSIDE THE CEREAL BOX ILLUSTRATION:\n",
"post": "\nOUTSIDE THE BOX:\nA Formica kitchen countertop with a checkered tablecloth and a glimpse of kitchen appliances."
},
{
"name": "Action Figure in Blister",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA classic action figure encased in a clear plastic blister pack. The cardboard backing is brightly colored and features bold graphics and text. The plastic blister is slightly warped and shows minor wear and tear, giving it a vintage feel. The background is a cluttered toy store shelf, with other action figures and collectibles visible.\nTHE ACTION FIGURE:\nA highly detailed, articulated action figure based on the following description:\n",
"post": "\nTHE BLISTER PACK:\nA clear plastic blister pack with a cardboard backing featuring vintage toy packaging design.\nTHE BACKGROUND:\nA cluttered toy store shelf with other action figures, collectibles, and price tags."
},
{
"name": "Antique Silver Coin",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA highly detailed, realistic rendering of an antique silver coin. The coin is heavily tarnished, with a dark patina that accentuates the relief details. The metal has a cool, silvery sheen, and the edges are slightly irregular. The background is a weathered wooden surface, adding to the coin's historical feel.\nTHE RELIEF ON THE COIN (FRONT):\nA deeply engraved relief portrait of a character, showing signs of wear and age. The details are intricate and expressive, capturing the character's personality.\nTHE RELIEF:\n",
"post": "\nTHE YEAR AND TEXT (AROUND THE RELIEF):\nThe year \"1888\" is engraved in a bold, sans-serif font along the lower edge of the coin. A circular inscription in a stylized, archaic typeface surrounds the portrait, reading \"REPUBLICA ARGENTINA\".\nTHE BACK OF THE COIN:\nA detailed depiction of a historical scene or symbol, such as a ship, a monument, or a coat of arms. The design is worn and faded, adding to the coin's antique appearance."
},
{
"name": "Little Music Box",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nAn open antique music box with a whimsical, illustrated style. The box is made of painted wood, featuring floral motifs and gilded details. The interior is brightly illuminated, creating a magical atmosphere.\nTHE MINIATURE FIGURE:\nA charming and stylized miniature figure, resembling a character from a classic fairy tale. The figure is slightly simplified, with exaggerated features and vibrant colors.\nTHE MINIATURE FIGURE:\n",
"post": "\nTHE ROTATING MECHANISM:\nThe figure is mounted on a white, circular platform, which is rotating. The rotating mechanism, with its colorful gears, is partially visible to its right.\nTHE SETTING:\nThe painted wood of the music box, the floral motifs, and the gilded details stand out prominently. The background is soft and dreamlike, with hints of a fairy-tale landscape."
},
{
"name": "Industrial Blueprint",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nAn extremely detailed technical blueprint, reminiscent of a 1950s industrial patent drawing. The background is a faded, aged blueprint paper with a subtle grid pattern. Lines are crisp and precise, rendered in varying shades of blue (cyan, dark blue, Prussian blue). The overall aesthetic is highly technical and schematic.\nTHE SUBJECT OF THE BLUEPRINT:\nA complex diagram of a mechanical being, meticulously labeled with technical annotations, dimensions, and material specifications. The subject is presented as a disassembled view, showcasing internal components and mechanisms.\nTHE DIAGRAM:\n",
"post": "\nADDITIONAL DETAILS:\nNumerous arrows indicating movement, force, or connection. Callout boxes with detailed descriptions of specific parts. A title block with a patent-style number and date."
},
{
"name": "Ancient Cave Painting",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nPrehistoric cave painting in earthy tones, created with natural pigments on a rough cave wall. The style is primitive and symbolic, depicting simplified animals and human figures. The painting is illuminated by the irregular light of a torch resting on the ground.\nIN THE ROCK WALL:\nA minimalist prehistoric painting without outlines\nINSIDE THE PREHISTORIC PAINTING:\n<!--\n",
"post": "\n-->\nOUTSIDE THE PREHISTORIC PAINTING:\nA cave with a rough rock wall, with visible cracks and fissures.\nThe cave is dark with dim lighting, but to the right, a tunnel can be seen, revealing the exit far in the distance under a blue sky."
},
{
"name": "Classical Mosaic",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA meticulously crafted image rendered in the style of a classical mosaic, reminiscent of ancient Roman or Byzantine art. The mosaic is composed of small, uniformly shaped tesserae (mosaic tiles) arranged in precise patterns. The overall effect is one of order, symmetry, and timeless elegance.\nTHE MOSAIC SURFACE:\nA smooth, flat plane entirely covered with tesserae. The surface is polished to a subtle sheen, reflecting light and enhancing the vibrancy of the colors.\nTHE MOSAIC DESIGN:\n",
"post": "\nTHE TESSERAE:\nSmall, square or rectangular tesserae made of glass, stone, or ceramic. The tesserae are carefully selected for their color and texture, and are arranged to create smooth gradients and intricate details.\nTHE PATTERN & SYMMETRY:\nHighly symmetrical and geometric patterns dominate the design. Repeating motifs and borders create a sense of balance and harmony. The composition is carefully planned to ensure visual coherence.\nTHE COLOR PALETTE:\nA refined color palette consisting of rich, saturated hues, such as deep blues, greens, reds, and golds. The colors are used to create contrast and highlight key elements of the design.\nTHE LIGHTING:\nEven, diffused lighting that illuminates the mosaic surface without creating harsh shadows. The light emphasizes the texture and reflectivity of the tesserae.\nTHE OVERALL FEEL:\nA sense of grandeur, sophistication, and historical significance. The image evokes the artistry and craftsmanship of ancient civilizations."
},
{
"name": "Folk-Art Mosaic",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA vibrant and whimsical image constructed entirely from colorful, irregularly shaped pieces of folk art mosaic tiles. The style evokes traditional crafts and a handmade aesthetic. The grout between the tiles is visible, adding texture and character. The overall impression is joyful and playful.\nTHE MOSAIC SURFACE:\nA flat plane covered in a dense arrangement of mosaic tiles. The tiles are made of various materials, including ceramic, glass, and stone, each with unique colors and patterns.\nTHE MOSAIC DESIGN:\n",
"post": "\nTHE TILE COLORS & PATTERNS:\nA rich palette of primary and secondary colors, with bold contrasts and playful combinations. The tiles feature floral motifs, geometric shapes, and stylized representations of natural elements.\nTHE OVERALL FEEL:\nA charming and rustic aesthetic, reminiscent of traditional folk art and handcrafted mosaics. The image exudes warmth, creativity, and a sense of nostalgia."
},
{
"name": "Hallway Frame Sign",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA two-part composition of a commercial interior. On the left, a plain wall houses a rectangular sign with a brushed metal frame and a clear acrylic panel. On the right, the image opens onto a well-lit corridor that fades into the distance, with people strolling by. The lighting is soft yet directional, allowing the metallic sheen of the frame and the depth of the corridor to be clearly visible.\nTHE LEFT WALL - METAL SIGN:\nThe wall is painted in a pastel color, providing a clean background for the sign. The frame is made of brushed steel or matte black iron, with slightly rounded corners and a subtle texture that catches the reflections of the ambient light. Inside the metal frame is an acrylic panel; its surface reflects a soft glow from the corridor lights, while a fine shadow outlines its perimeter, emphasizing its depth.\nINSIDE THE METAL SIGN:\nThere is a minimalist, illustration that serves as a warning to passersby.\nWITHIN THE MINIMALIST ILLUSTRATION:\n<!--\n",
"post": "\n-->\nRIGHT SIDE - ESTABLISHMENT HALLWAY:\nThe hallway extends from the base of the sign to the vanishing point. The floor is made of glazed tiles that reflect a soft glow. The ceiling houses recessed lighting that bathes the hallway in a warm, even light, creating gentle gradients of light and shadow that enhance the sense of depth.\nTHE PHOTO:\nThe final image was captured from a slightly tilted perspective that places the sign in the foreground on the left, with the hallway receding on the right. The depth of field is moderate, keeping both the details of the sign and the main lines of the hallway sharp, while a subtle vignette directs attention to the center, where the user-provided artwork is located within the acrylic panel."
},
{
"name": "Miniature Diorama",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA meticulously crafted miniature diorama inside a weathered wooden box.\nTHE DIORAMA:\nA miniature scene recreated with incredible attention to detail. Everything is tiny, with hand-painted wooden objects and realistic textures. The objects are detailed and antique, all made of wood.\nINSIDE THE DIORAMA SCENE:\n",
"post": "\nOUTSIDE THE DIORAMA SCENE:\nA weathered wooden box with a hinged lid. The inside of the box is lined with red velvet.\nLIGHTING:\nSoft, diffused lighting illuminates the diorama and creates a sense of intimacy."
},
{
"name": "Paper Cut Diorama",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA photo of a diorama constructed entirely from layers of intricately cut paper. The scene is viewed through a frame, the simple and elegant frame surrounding the diorama. The lighting is soft and diffused, highlighting the delicate details of the paper cutouts.\nTHE PAPER SCENE:\nA detailed, layered representation rendered in a paper-cut style. Multiple layers of paper create depth and dimension.\nINSIDE YOUR FLAT PAPER SCENE:\nEverything is flat: the background is flat, the characters are flat, there's no depth, everything is as thin as a layer of paper. The detailed background flat and drawn.\nINSIDE YOUR FLAT PAPER SCENE:\n",
"post": "\nTHE LAYERS:\nForeground elements are closer to the viewer, while background elements are farther away. Each layer is carefully cut and assembled to create a cohesive scene.\nSOME DETAILS:\nIntricate patterns, delicate textures, and a handcrafted feel. The paper has a slightly textured appearance, enhancing the tactile quality of the diorama.\nTHE PHOTO:\nA close-up of the paper cutout diorama, highlighting each of its layers and the masterfully crafted details."
},
{
"name": "Marble Statue",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA carefully curated museum exhibit. The exhibit is dimly lit, with spotlights highlighting the centerpiece. The atmosphere is reverent and scholarly.\nTo the left, a stunning hyperrealistic statue crafted from marble is visible. The statue features a dramatic pose that captures the essence of the subject.\nWITHIN THE HYPERREALIST STATUE:\n",
"post": "\nOUTSIDE THE HYPERREALIST STATUE:\nA polished pedestal, a velvet rope, and an information plaque describing the artwork.\nTHE BACKGROUND:\nA large museum hall with ornate architecture and other exhibits visible to the right, as well as some elegantly dressed people walking in the distance."
},
{
"name": "Stamp",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA photo of a letter, the photo is centered on the stamp, which occupies most of the frame.\nWITHIN THE STAMP:\nThere is a monochromatic illustration featuring a striking monochromatic stamp style, characterized by bold, simplified black outlines and minimal use of pointillism or halftone patterns. The aesthetic evokes the precision of a linocut or woodcut, in high-contrast black and white. The style is highly minimalist and iconographic, focusing on essential forms and symbolic representation. Faces, when presented, are reduced to their most recognizable features, resembling stylized icons rather than realistic portraits. The ample white space is a defining element, creating a sense of elegance and emphasizing the symbolic weight of the images. At the top right, it says \"$1,\" which is the stamp's value.\nWITHIN THE ILLUSTRATION:\n",
"post": "\nOUTSIDE THE ILLUSTRATION:\nA macro photo of the stamp affixed to the letter shows the texture and imperfections of the stamp as well as the texture of the yellowed and worn paper of the letter."
},
{
"name": "Victorian Scrapbook Page",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA beautifully decorated page from a Victorian-era scrapbook. The page is filled with pressed flowers, delicate lace, handwritten calligraphy, and intricate paper cutouts. The overall aesthetic is romantic and nostalgic.\nIN THE CENTER OF THEPAGE:\nA rectangular faded illustration, carefully mounted. The faded illustration is a portal.\nBEHIND THE PORTAL:\n<!--\n",
"post": "\n-->\nOUTSIDE THE PORTAL:\nPressed flowers, delicate lace, handwritten calligraphy, intricate paper cutouts, and colorful ribbons.\nTHE SCRAPBOOK PAGE:\nA textured, aged paper page with visible wear and tear. The page is adorned with a variety of embellishments and mementos."
},
{
"name": "Vintage Reel Frame",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nClose-up of a classic 35mm nitrate film from the 1930s. The film runs horizontally across the image, with the perforation sequence above and below. The film is laid on a wooden table. The film strip is partially unwound, exposing individual frames. In this extreme close-up, the rectangular frame of the strip fills the entire image.\nINSIDE A FILM FRAME:\n<!--\n",
"post": "\n-->\nOUTSIDE THE FILM FRAME:\nA wooden table with shadows and light flares projected onto the table."
},
{
"name": "Holographic Device",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA futuristic holographic projector sits atop a sleek, polished pedestal in a minimalist showroom. The projection emits a semi-transparent, three-dimensional image that floats in the air, illuminated by a soft, bluish-white light. The surrounding space is clean, with subtle reflections on the floor.\nThe holographic screen is positioned at a slightly elevated angle, capturing the floating image.\nTHE HOLOGRAPHIC PROJECTOR\nIt's a heavy metal tube with an industrial look and the appearance of a radiator. A powerful red reflector shines from its top side.\nINSIDE YOUR HOLOGRAPHIC IMAGE:\nThe image features blue tones with intensely illuminated light blue lines and wireframe details of light blue lines.\nINSIDE YOUR HOLOGRAPHIC IMAGE:\n<!--\n",
"post": "\n-->\nOUTSIDE YOUR HOLOGRAPHIC IMAGE:\nThe showroom dimly lit by the hologram's blue light."
},
{
"name": "Coffee-Cup Art Print",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nAn extreme close-up of the foam on a freshly brewed espresso cup. The surface of the foam is printed with a small hand-drawn illustration. The coffee cup is a classic white porcelain mug with a glossy rim, placed on a rustic wooden table.\nINSIDE THE FOAM:\nA draw in shades of brown and white with blurred lines and foam relief.\nINSIDE YOUR FOAM BROWN DRAW:\n<!--\n",
"post": "\n-->\nOUTSIDE THE ARTWORK:\nThe coffee is in the foreground with a shallow depth of field, shot taken from a slight top-down angle. The foam is subtly depicted against the dark coffee, while the background shows a blurred cafe scene: a wooden table, other cups and a black background in the top. Light steam rises gently, creating a cozy atmosphere."
},
{
"name": "Cassette Tape Sleeve",
"cat": "Wild",
"pre": "SCENE:\nA mixtape from the 1990s lying on a rug in a clutter-filled bedroom. The case stands upright and is made of clear plastic.\nINSIDE THE SLEEVE:\n<!--\n",
"post": "\n-->\nOUTSIDE THE SLEEVE:\nA nostalgic photograph of the cassette tape lying on a faded carpet. The word \"CASSETTE\" can be seen written on it. The image captures the handwritten typography, and faint tape-track reflections. Ambient teal-green bedroom light creates a soft, retro vibe."
},
{
"name": "Vintage Postcard",
"cat": "Wild",
"pre": "YOUR CONTEXT:\nA postcard printed in the 1950s depicts a charming coastal village. The card is slightly creased, its edges softened by years of handling, and the paper has a warm, yellowed patina.\nINSIDE ARTWORK ON THE FRONT OF THE POSTCARD:\nImage printed in faded colors, blurry and of low quality.\nINSIDE ARTWORK ON THE FRONT OF THE POSTCARD:\n<!--\n",
"post": "\n-->\nOUTSIDE ARTWORK:\nA flat, highly detailed photograph of the postcard on a textured linen surface. The postcard's border is embossed, and a handwritten address line (blurred for privacy) is visible. Soft natural light from a nearby window creates gentle shadows that accentuate the nostalgic feel."
},
{
"name": "Crime Game Loading Screen",
"cat": "Illustration",
"pre": "A modern open-world crime video game loading-screen illustration of ",
"post": ". Art style: bold clean ink outlines, cel-shaded skin and clothes with hard-edged graphic shadows and highlights, slightly stylised realistic faces and proportions, a saturated sunset palette of lavender, violet, magenta, peach and warm gold with teal accents, a hazy Miami-style Art Deco skyline and palm trees, strong warm rim light, poster-like composition, highly detailed hand-drawn digital painting."
},
{
"name": "Crime Game Character Portrait",
"cat": "Illustration",
"pre": "A waist-up character portrait for a modern open-world crime video game, loading-screen art of ",
"post": ". Confident pose, looking at the viewer, plain hazy sunset background with the city skyline. Art style: bold clean ink outlines, cel-shaded skin and clothes with hard-edged graphic shadows and highlights, slightly stylised realistic faces and proportions, a saturated sunset palette of lavender, violet, magenta, peach and warm gold with teal accents, a hazy Miami-style Art Deco skyline and palm trees, strong warm rim light, poster-like composition, highly detailed hand-drawn digital painting."
},
{
"name": "Winter Storm",
"cat": "Effects",
"pre": "A photograph of ",
"post": "In a heavy snowfall. Snow settles naturally on the top of the head and hair, the shoulders, the folds of the clothes and every upward-facing edge, with fine frost on hair and fabric; big soft flakes fall through the air, some close to the camera and blurred; fresh snow on the ground, drifting mist. Cold blue-grey winter light, breath visible in the cold air, a cool desaturated grade with icy highlights."
},
{
"name": "Soaked in Rain",
"cat": "Effects",
"pre": "A photograph of ",
"post": "Soaking wet in a heavy downpour. Hair wet and clumped, skin glistening with beads of water, clothes dark and clinging, water dripping from the chin, fingers and hems; on things, beading droplets and water running off every edge. Long streaks of rain, splashes and reflective puddles, moody low light with wet specular highlights, a cool high-contrast grade."
},
{
"name": "Concept Art Painting",
"cat": "Effects",
"pre": "A loose, expressive digital concept-art painting of ",
"post": "The face and main forms painted with confident detail, dissolving towards the edges into broad brushstrokes, palette-knife marks, paint splatters and drips over an abstract painterly wash; rich heightened colour with warm and cool accents, strong contrast between finished and unfinished areas."
},
{
"name": "Dramatic Light",
"cat": "Effects",
"pre": "A dark, dramatic cinematic photograph of ",
"post": "A single beam of light from directly above through faint haze with dust in it, bright rim light on the head and shoulders, everything else falling into deep shadow on a near-black background; desaturated steel-blue and teal grade, deep blacks, crisp texture, subtle film grain, like a film poster."
},
{
"name": "Double Exposure",
"cat": "Effects",
"pre": "An artistic double-exposure photograph: the silhouette of ",
"post": "Filled with a misty pine forest on a mountainside that blends through the body and the back of the head while the face shows softly through; a clean off-white background, treetops and mist breaking past the outline, a muted natural palette, soft fog, fine grain, minimal and elegant."
},
{
"name": "Made of Water",
"cat": "Effects",
"pre": "A high-speed photograph of ",
"post": " made entirely of clear liquid water: the same shape and details formed from rippling glassy water with bright highlights and refraction, splashes and droplets bursting off the edges, streams flowing from the limbs, small bubbles inside; a deep teal-blue underwater background with soft light rays."
},
{
"name": "Apocalypse",
"cat": "Effects",
"pre": "A cinematic photograph of ",
"post": "After an apocalypse: cracked, scorched and crumbling, broken windows, rust and soot, rubble on the ground, small fires with glowing embers, thick black smoke rising into an ash-grey sky; a desaturated dirty teal-and-orange grade."
},
{
"name": "Made of Smoke",
"cat": "Effects",
"pre": "A dark ethereal photograph of ",
"post": " formed from swirling wisps of glowing smoke: the face clearest, the body built from soft smoke following its forms, the edges curling away into drifting smoke trails and haze; a near-black background, the smoke lit cool white and pale blue from within."
},
{
"name": "Pencil Sketch",
"cat": "Effects",
"pre": "A hand-drawn graphite pencil sketch of ",
"post": "On textured white drawing paper: confident loose outlines, parallel diagonal hatching and cross-hatching for the shading, a few construction lines, fading out unfinished towards the edges; pencil grey only, the paper's grain visible."
},
{
"name": "Watercolour Splash",
"cat": "Effects",
"pre": "A vivid watercolour painting of ",
"post": "On white watercolour paper: the face and forms in careful transparent washes, the colours bleeding outward into loose blooms, splashes, drips and spatters, the background mostly bare paper with soft stains, hard wet edges, luminous colour."
},
{
"name": "Cartoon Portrait",
"cat": "Effects",
"pre": "A bold digital cartoon portrait of ",
"post": "Clean confident dark outlines, smooth cel shading with soft gradients, gently exaggerated proportions with expressive eyes and smile, crisp highlights, rich saturated colours, a simple bold flat-colour background with a few graphic shapes; polished professional illustration."
},
{
"name": "Torn Paper",
"cat": "Effects",
"pre": "A torn-paper photo effect of ",
"post": "The picture printed in black and white halftone on a sheet of paper, a large hole ripped through the middle where the face and upper body show in full rich colour, rough white fibrous torn edges curling outward with soft shadows."
},
{
"name": "Sparkler Light Trail",
"cat": "Effects",
"pre": "A long-exposure night photograph of ",
"post": "With a bright sparkler light-painting trail looping and spiralling around them, passing behind and in front: a continuous line of warm golden light, white-hot at its core, a soft glow and countless tiny sparks flying off it; the dark scene softly lit by the trail."
},
{
"name": "Fluted Glass",
"cat": "Effects",
"pre": "A minimal, elegant photograph of ",
"post": "Partly seen through a large panel of fluted reeded glass (a tall rounded rectangle turned 45 degrees): through the glass the picture splits into thin vertical ribs that smear the colours into streaks with faint colour fringes, outside it everything sharp, the face clear."
},
{
"name": "Line Engraving",
"cat": "Effects",
"pre": "A detailed banknote-style line engraving of ",
"post": "Fine parallel lines that thicken in the shadows and curve with the forms, cross-hatching in the darkest areas, black ink on off-white paper, crisp and precise like an antique engraved portrait."
},
{
"name": "Dispersion",
"cat": "Effects",
"pre": "A dramatic dispersion photo effect of ",
"post": "One side breaks apart into thousands of small shards, flakes and dust, each carrying the colour of the part it came from, streaming away as if blown by a strong wind, the face and main form whole and sharp; a dark, softly graded background, dramatic side light on the flying pieces."
},
{
"name": "Glitch",
"cat": "Effects",
"pre": "A digital glitch photo of ",
"post": "Red and blue channels split and offset with colour fringes on every edge, horizontal bands torn and shifted sideways, smeared pixel streaks, scan lines, blocky compression and noise, like a damaged video signal; cold high-contrast grade with magenta and cyan accents."
},
{
"name": "Marble Statue",
"cat": "Effects",
"pre": "A photograph of a lifelike white marble sculpture of ",
"post": "Everything carved in polished white marble with fine grey veins, sculpted locks of hair, carved folds of cloth, smooth blank eyes; on a stone plinth in a quiet museum gallery under a soft skylight, deep shadows."
},
{
"name": "Low-Poly",
"cat": "Effects",
"pre": "Low-poly geometric art of ",
"post": "Built from flat triangles of solid colour, small and dense on the face and details, larger elsewhere, faint wireframe lines between facets, some triangles breaking away and floating off; a clean gradient background, crisp vector art."
},
{
"name": "Ink Splash",
"cat": "Effects",
"pre": "A bold black ink artwork of ",
"post": "Sharp confident ink on the face and main features, dissolving towards the edges into big splashes, wet brush strokes, drips and spatter, grey washes where the ink was thinned, bare white paper; high contrast black and white."
},
{
"name": "Stained Glass",
"cat": "Effects",
"pre": "A photograph of a stained-glass window showing ",
"post": "Pieces of coloured glass joined by dark lead lines, small pieces with painted shading for the face, larger decorative pieces for the clothes and background, glowing jewel colours lit from behind, bubbles and ripples in the glass, an arched stone frame."
},
{
"name": "Hologram",
"cat": "Effects",
"pre": "A projected hologram of ",
"post": "Semi-transparent glowing cyan light, fine horizontal scan lines, a few flickering shifted bands, glowing particles drifting off the edges, a cone of light rising from a small round projector; a dark high-tech room lit blue by the glow."
},
{
"name": "Embroidered Patch",
"cat": "Effects",
"pre": "A close-up photograph of an embroidered patch of ",
"post": "Dense satin stitches following the shapes and shading, blended rows of thread in close shades, soft thread sheen, a raised edge and a thick stitched border, sewn on denim, soft light showing every thread."
},
{
"name": "Layered Paper Cut",
"cat": "Effects",
"pre": "A layered paper-cut artwork of ",
"post": "Cut from many stacked sheets of thick matte coloured paper, clean smooth curved edges, soft real shadows between the layers giving depth, fine paper texture, a few curled quilled details, softly lit from one side; a handmade paper diorama."
}
];

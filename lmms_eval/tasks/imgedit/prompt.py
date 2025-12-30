IMGEDIT_PROMPTS = {
    "replace": """
You are a data rater specializing in grading image replacement edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the replacement editing effect on a 5-point scale from three perspectives:

Prompt Compliance
1  Target not replaced, or an unrelated object edited.
2  Only part of the target replaced, or wrong class/description used.
3  Target largely replaced but other objects altered, remnants visible, or count/position clearly wrong.
4  Correct object fully replaced; only minor attribute errors (colour, size, etc.).
5  Perfect replacement: all and only the specified objects removed; new objects' class, number, position, scale, pose and detail exactly match the prompt.

Visual Naturalness
1  Image heavily broken or new object deformed / extremely blurred.
2  Obvious seams, smears, or strong mismatch in resolution or colour; background not restored.
3  Basic style similar, but lighting or palette clashes; fuzzy edges or noise are noticeable.
4  Style almost uniform; tiny edge artefacts visible only on close inspection; casual viewers see no edit.
5  Completely seamless; new objects blend fully with the scene, edit area undetectable.

Physical & Detail Integrity
1  Floating, interpenetration, severe perspective/light errors; key original elements ruined; background heavily warped.
2  Missing shadows/occlusion; large background shifts or holes.
3  Lighting, perspective and contact surfaces mostly correct; small but tolerable errors; background adjusted locally.
4  New objects interact realistically with scene (shadows, reflections, texture) and preserve existing details; background change minimal.
5  Physically flawless and enhances realism: accurate highlights, shadows, reflections, ambient effects; background untouched.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical & Detail Integrity: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "add": """
You are a data rater specializing in grading image addition edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the added object(s) on a 5-point scale from three perspectives:

Prompt Compliance
1  Nothing added or the added content is corrupt.
2  Added object is a wrong class or unrelated to the prompt.
3  Correct class, but key attributes (position, colour, size, count, etc.) are wrong.
4  Main attributes correct; only minor details off or 1-2 small features missing.
5  Every stated attribute correct and scene logic reasonable; only microscopic flaws.

Visual Naturalness
1  Image badly broken or full of artefacts.
2  Obvious paste marks; style, resolution, or palette strongly mismatch.
3  General style similar, but lighting or colours clearly clash; noticeable disharmony.
4  Style almost uniform; small edge issues visible only when zoomed.
5  Perfect blend; no visible difference between added object and original image.

Physical & Detail Coherence
1  Severe physical errors (floating, wrong perspective/light); key original elements blocked; background heavily distorted.
2  Contact or occlusion handled poorly; minor background shifts, jaggies or noise; background visibly changed.
3  Lighting, perspective, and contact mostly correct; remaining flaws small and acceptable; limited background change.
4  Shadows, reflections, and material response believable; no loss of original detail; background changes are minute.
5  Added object enhances overall realism: precise highlights, shadows, ambient effects; background essentially untouched.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical & Detail Coherence: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "adjust": """
You are a data rater specializing in grading attribute alteration edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the attribute change on a 5-point scale from three perspectives:

Prompt Compliance
1  Target not adjusted, wrong object touched, or geometry changed.
2  Right object but wrong attribute value/direction; only part edited; other objects also altered; slight stretch/crop.
3  Mainly correct object and attribute, yet large hue/brightness/texture error; minor collateral edits; visible jaggies/distortion.
4  All requested objects adjusted, only their attributes changed; shape kept; small inaccuracy in colour, material or amount.
5  Exactly and only the requested objects adjusted; colour, material, gloss etc. match the prompt perfectly; shape 100% intact; zero unintended edits.

Visual Seamlessness
1  Massive colour spill, mosaics or heavy noise; image nearly unusable.
2  Clear smears/bleeding on edges; abrupt resolution or tone shift; highlights/shadows clipped; background gaps.
3  Overall palette OK but local tone or grain conflicts; soft edges; noticeable disharmony.
4  Style unified, transitions smooth; only slight edge artefacts visible when zoomed.
5  No detectable edit traces; colours/materials fuse with scene lighting; edit area practically invisible.

Physical & Detail Fidelity
1  Object floating, interpenetrating, or severe perspective/light mismatch; background badly warped.
2  Missing shadows/highlights; wrong reflection direction; background visibly discoloured or distorted.
3  Light, perspective and contact surface largely correct; minor acceptable flaws; background only locally affected.
4  Adjusted material interacts believably with scene; shadows, highlights, reflections handled well; original details preserved.
5  High physical realism: fine micro-highlights, diffuse bounce, subsurface effects present; overall scene realism improved.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Seamlessness: A number from 1 to 5.
Physical & Detail Fidelity: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "remove": """
You are a data rater specializing in grading object removal edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the removal quality on a 5-point scale from three perspectives:

Prompt Compliance
1  Nothing removed, or an unrelated object edited.
2  Target only partly removed, or a different instance/class deleted, or another object appears in the gap.
3  Target mostly removed but extra objects also deleted, or fragments of the target remain.
4  Only the specified objects removed, but a few tiny/background items deleted by mistake, or the count is wrong.
5  Perfect: all and only the requested objects removed; every other element untouched.

Visual Naturalness
1  Image badly broken (large holes, strong artefacts).
2  Clear erase marks; colour/resolution mismatch; background not restored.
3  General look acceptable yet lighting/colour/style still clash; blur or noise visible.
4  Style consistent; minor edge issues visible only when zoomed.
5  Seamless: removal is virtually impossible to spot.

Physical & Detail Integrity
1  Severe physical errors (floating items, wrong perspective/light); key scene elements damaged; background heavily warped.
2  Large un-filled gaps or obvious background shifts.
3  Lighting, perspective and contacts mostly correct; flaws small and tolerable; background adjusted locally.
4  Background reconstruction clean; existing details preserved; only minute changes outside the removal area.
5  Physically flawless and even enhances realism: accurate light/shadow/texture infill, high-quality micro-details.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical & Detail Integrity: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "style": """
You are a data rater specializing in grading style transfer edits. You will be given an input image, a reference style, and the styled result. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:

Style Fidelity
1  Target style absent or clearly wrong.
2  Style shows in a few areas only, or mixed with unrelated styles.
3  Key traits (palette, brushwork, texture) present but patchy or inconsistent.
4  Style reproduced across almost the whole image; only small local mismatches.
5  Full, faithful transfer: colour, texture, brushwork, lighting all match the exemplar over the entire image.

Content Preservation
1  Major objects or layout lost/distorted; original scene barely recognisable.
2  Main subject recognisable, but size, perspective or key parts clearly wrong/missing.
3  Overall structure correct; some local warping or minor omissions.
4  Nearly all geometry intact; only slight, non-distracting deformation.
5  All objects and spatial relations kept; only stylistic, harmless distortion.

Rendering Quality
1  Heavy noise, banding, pixel damage or blur; image unusable.
2  Visible seams, aliasing, colour drift; low resolution or chaotic strokes.
3  Moderate quality: local blur/noise/texture breaks, but generally acceptable.
4  Sharp, coherent strokes; tiny artefacts visible only when zoomed.
5  High resolution, no artefacts; strokes, textures and colour transitions look fully natural.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Style Fidelity: A number from 1 to 5.
Content Preservation: A number from 1 to 5.
Rendering Quality: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the input, reference style, and styled output image:
""",
    "action": """
You are a data rater specializing in grading action or expression change edits. You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the motion or expression change on a 5-point scale from three perspectives:

Action / Expression Fidelity
1  No visible change, or wrong action / expression.
2  Partial or clearly incorrect pose; only some body parts change; expression direction wrong.
3  Main idea present but details off (angle, side, intensity, missing gesture).
4  Requested pose / expression achieved with just minor inaccuracy (small angular drift, timing nuance).
5  Exact match to prompt: every limb, gesture, and facial muscle aligns with the described action.

Identity Preservation
1  Person unrecognisable; face or body replaced.
2  Strong drift: key facial features, hairstyle or clothing heavily altered.
3  Mostly same identity; moderate changes in some features but still recognisable.
4  Identity clearly the same; only subtle stylisation or lighting differences.
5  Perfect preservation of face, hairstyle, skin tone, clothing and accessories.

Visual & Anatomical Coherence
1  Severe artifacts: broken or duplicated limbs, extreme distortion, heavy noise/blur.
2  Noticeable cut-out halos, proportion errors, lighting or perspective clearly off.
3  Generally plausible; minor joint or shading issues; small noise/blur acceptable.
4  Clean render; anatomy, lighting, depth and edges consistent; flaws only on close inspection.
5  Flawless realism or stylistic coherence; perfect anatomy, lighting, shadows and texture continuity.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Action Fidelity: A number from 1 to 5.
Identity Preservation: A number from 1 to 5.
Visual & Anatomical Coherence: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "extract": """
You are a data rater specializing in grading object cut-out quality. You will be given an image with the object extracted on a white background. Your task is to evaluate the cut-out accuracy on a 5-point scale from three perspectives:

Object Selection & Identity
1  Wrong object or multiple objects extracted.
2  Correct class but only part of the object, or obvious intrusions from other items.
3  Object largely correct yet small pieces missing / extra, identity still recognisable.
4  Full object with clear identity; only tiny mis-crop (e.g., tip of antenna).
5  Exact requested object, complete and unmistakably the same instance (ID).

Mask Precision & Background Purity
1  Large background remnants, holes in mask, or non-white backdrop dominates.
2  Noticeable jagged edges, colour fringes, grey/colour patches in white area.
3  Acceptable mask; minor edge softness or faint halo visible on close look.
4  Clean, smooth edges; white (#FFFFFF) background uniform, tiny artefacts only when zoomed.
5  Crisp anti-aliased contour, zero spill or halo; backdrop perfectly pure white throughout.

Object Integrity & Visual Quality
1  Severe blur, compression, deformation, or missing parts; unusable.
2  Moderate noise, colour shift, or slight warping; details clearly degraded.
3  Overall intact with minor softness or noise; colours mostly preserved.
4  Sharp detail, accurate colours; negligible artefacts.
5  Pristine: high-resolution detail, true colours, no artefacts or distortion.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Object Identity: A number from 1 to 5.
Mask Precision: A number from 1 to 5.
Visual Quality: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below is the extracted object image:
""",
    "background": """
You are a data rater specializing in grading background editing. You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:

Instruction Compliance
1  No change, or background unrelated to prompt, or foreground also replaced/distorted.
2  Background partly replaced or wrong style/content; foreground noticeably altered.
3  Main background replaced but elements missing/extra, or faint spill onto subject edges.
4  Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).
5  Background exactly matches prompt (content, style, placement); all foreground pixels untouched.

Visual Seamlessness (Edge & Texture Blend)
1  Large tearing, posterisation, extreme blur/noise; edit area obvious at a glance.
2  Clear cut-out halos, colour-resolution gap, or heavy smudge strokes.
3  Blend acceptable but visible on closer look: slight edge blur, grain or palette shift.
4  Nearly invisible seams; textures and sharpness aligned, only minor issues when zoomed in.
5  Indistinguishable composite: edges, textures, resolution and colour grading perfectly continuous.

Physical Consistency (Lighting, Perspective, Depth)
1  Severe mismatch: wrong horizon, conflicting light direction, floating subject, warped geometry.
2  Noticeable but not extreme inconsistencies in light, shadows or scale; depth cues off.
3  Overall believable; small errors in shadow length, perspective or ambient colour.
4  Lighting, scale, depth, and camera angle well matched; only subtle discrepancies.
5  Physically flawless: foreground and new background share coherent light, shadows, reflections, perspective and atmospheric depth, enhancing overall realism.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Instruction Compliance: A number from 1 to 5.
Visual Seamlessness: A number from 1 to 5.
Physical Consistency: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "compose": """
You are a data rater specializing in grading hybrid image edits (involving multiple operations on multiple objects). You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the overall editing quality on a 5-point scale from three perspectives:

Instruction Compliance
1  Neither object nor operations match the prompt; wrong items edited or shapes distorted.
2  Only one object correctly edited, or both edited but with wrong/partial operations; collateral changes to other items.
3  Both target objects touched, each with the requested operation broadly correct but missing details (e.g., wrong colour value, incomplete removal).
4  Both objects receive the exact operations; tiny deviations in amount, position, or parameter. No unintended edits elsewhere.
5  Perfect execution: each object fully reflects its specified operation, all other scene elements untouched.

Visual Naturalness (Seamlessness)
1  Large artefacts, obvious cut-outs, heavy blur/noise; edits conspicuous at a glance.
2  Clear edge halos, colour or resolution mismatch, awkward scaling.
3  Acceptable but visible on close look: slight edge softness, minor palette or focus shift.
4  Edits blend smoothly; seams hard to spot, textures and sharpness largely consistent.
5  Indistinguishable composite: colour grading, grain, resolution and style fully match the original image.

Physical Consistency & Fine Detail
1  Severe lighting/perspective mismatch, missing or wrong shadows; objects appear floating or warped.
2  Noticeable but tolerable inconsistencies in illumination, scale, or depth cues.
3  Generally plausible; small errors in shadow length, reflection angle, or texture alignment.
4  Lighting, perspective, and material response closely match; only subtle flaws visible when zoomed.
5  Physically flawless: shadows, highlights, reflections, depth and texture perfectly integrated, enhancing overall realism.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Instruction Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical Consistency & Fine Detail: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
}

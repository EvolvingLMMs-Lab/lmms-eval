# Visual CoT 任务 Prompt 汇总

## 1. MMSI-Bench

### 1.1 直接生成

```
{pre_prompt}{question}{post_prompt}
```

其中 `post_prompt` 通常为：
```
Based on your visual observation, answer with the option's letter from the given choices directly. Enclose the option's letter within ``.
```

### 1.2 Visual CoT

**Stage 1 Generation Prompt（在 YAML 中定义）：**

| 子任务 | Generation Prompt |
|--------|-------------------|
| Attribute (Appr.) | Create a visualization that highlights and labels the visual appearance attributes (color, shape, texture, orientation, count) in the scene. Use annotations, bounding boxes, and labels to make object features and counts clearly visible. |
| Attribute (Meas.) | Create a visualization that highlights and labels measurement attributes in the scene. Use annotations to show sizes, distances, proportions, and scales clearly. |
| Motion (Obj.) | Create a visualization that tracks and highlights object motion in the scene. Use arrows, trajectories, and annotations to show movement direction, speed, and path of objects. |
| Motion (Cam.) | Create a visualization that illustrates camera motion characteristics. Use arrows and annotations to show camera movement type (pan, tilt, zoom, dolly), direction, and magnitude. |
| MSR | Create a visualization that highlights spatial relationships across the images. Use annotations, connecting lines, and labels to show relative positions, orientations, and spatial correspondence between elements. |

**Stage 2 Question Prompt：**
```
You are given the original image(s) and a visualization highlighting [attributes/motion/spatial relationships].
Use both to analyze [specific aspects].

{question}

Based on your visual observation, answer with the option's letter from the given choices directly.
Enclose the option's letter within ``.
```

---

## 2. VisualPuzzles

### 2.1 直接生成

```
Question: {question}
Options:
(A) {option_A}
(B) {option_B}
(C) {option_C}
(D) {option_D}
Solve the multiple-choice question and then answer with the option letter from the given choices.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.
Think step by step before answering.
```

### 2.2 Visual CoT

**Stage 1 Generation Prompt：**

| 子任务 | Generation Prompt |
|--------|-------------------|
| Algorithmic | You are given an algorithmic reasoning puzzle. Analyze the puzzle and create a helpful visualization.<br><br>{question with options}<br><br>Your task:<br>1. Identify any numerical sequences, patterns, or computational rules in the puzzle<br>2. Create a diagram that clearly shows:<br>   - The step-by-step computation or transformation process<br>   - Arrows or annotations showing how numbers/symbols change<br>   - The mathematical relationship or formula discovered<br>   - Highlighted patterns (e.g., +2, ×3, alternating, etc.)<br>3. Label each step of the algorithm clearly<br><br>Generate a clear diagram that reveals the underlying algorithmic pattern. |
| Analogical | You are given an analogical reasoning puzzle (A is to B as C is to ?).<br><br>{question with options}<br><br>Your task:<br>1. Identify the transformation relationship between the first pair of elements<br>2. Create a diagram that clearly shows:<br>   - What changes occur (rotation, reflection, color change, size change, addition/removal of elements)<br>   - Arrows indicating the direction and type of transformation<br>   - Labels describing each transformation (e.g., "rotate 90°", "invert colors", "add dot")<br>   - Apply the same transformation to show what the answer should look like<br>3. Make the analogy relationship visually explicit<br><br>Generate a diagram that reveals the transformation pattern between pairs. |
| Deductive | You are given a deductive reasoning puzzle that requires logical inference.<br><br>{question with options}<br><br>Your task:<br>1. Identify the given premises, rules, or constraints in the puzzle<br>2. Create a diagram that clearly shows:<br>   - All given conditions/rules listed clearly<br>   - A logical flowchart or inference chain<br>   - Step-by-step deduction from premises to conclusion<br>   - Elimination of incorrect possibilities<br>   - The logical path leading to the answer<br>3. Use arrows to show the deduction flow<br><br>Generate a logical inference diagram that traces the reasoning path. |
| Inductive | You are given an inductive reasoning puzzle that requires pattern recognition.<br><br>{question with options}<br><br>Your task:<br>1. Observe the sequence of examples and identify the underlying pattern<br>2. Create a diagram that clearly shows:<br>   - The repeating elements or motifs highlighted/circled<br>   - The progression rule (what changes from one step to the next)<br>   - Annotations showing the pattern cycle or growth rule<br>   - A prediction of what comes next based on the pattern<br>   - Color-coding or numbering to show pattern repetition<br>3. Make the inductive pattern visually obvious<br><br>Generate a diagram that highlights the repeating pattern and predicts the next element. |
| Spatial | You are given a spatial reasoning puzzle involving 3D visualization or transformations.<br><br>{question with options}<br><br>Your task:<br>1. Analyze the spatial transformation required (rotation, folding, unfolding, different viewpoint)<br>2. Create a diagram that clearly shows:<br>   - The object from multiple angles if rotation is involved<br>   - Step-by-step folding/unfolding process if applicable<br>   - Arrows indicating rotation direction and degree<br>   - Reference points or markers to track orientation<br>   - The resulting shape after transformation<br>3. Add axis lines or reference frames to clarify spatial orientation<br><br>Generate a multi-view or step-by-step transformation diagram. |

**Stage 2 Question Prompt：**
```
{question with options}

You are given TWO images:
1) ORIGINAL PUZZLE: The {category} reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the {category-specific description}

Use the auxiliary diagram to understand the {reasoning type}, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly.
```

---

## 3. Geometry3K

### 3.1 直接生成

```
{problem}

Instructions:
1. Carefully analyze the geometry diagram shown above.
2. Read the problem statement and identify what needs to be found.
3. Show your step-by-step solution with clear reasoning.
4. Include all intermediate calculations.
5. State the final answer clearly at the end.

Please solve this problem step by step.
```

### 3.2 Visual CoT

**Stage 1 Generation Prompt：**
```
You are given a geometry problem with a diagram. Analyze the problem and create an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {problem}

Instructions:
1. KEEP all original elements exactly as they are (all points, lines, labels, and measurements)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines):
   - Perpendicular lines from center to chords
   - Extended lines if needed
   - Angle bisectors, midpoints, or other helpful constructions
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original with extra auxiliary lines drawn on top

Generate an enhanced diagram that preserves the original and adds helpful auxiliary constructions.
```

**Stage 2 Question Prompt：**
```
{problem}

You are given TWO images:
1) ORIGINAL DIAGRAM: The geometry problem as given
2) AUXILIARY DIAGRAM: The same diagram with auxiliary constructions (extra lines) added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added
2. Use these auxiliary lines to identify key geometric relationships (perpendiculars, congruent segments, etc.)
3. Apply relevant theorems (Pythagorean theorem, chord properties, etc.)
4. Show your step-by-step solution with clear calculations
5. State the final numerical answer

Solve this problem step by step.
```

---

## 4. AuxSolidMath

### 4.1 直接生成

```
You are given a solid geometry problem with a 3D diagram.

Problem: {question}

Instructions:
1. First, carefully analyze the 3D diagram and identify what auxiliary lines (辅助线) need to be drawn to solve this problem. Common auxiliary constructions in solid geometry include:
   - Connecting points to form line segments
   - Drawing perpendiculars from a point to a plane or line
   - Finding midpoints and connecting them
   - Extending lines to find intersections
   - Drawing parallel lines through specific points
   - Constructing cross-sections

2. Clearly state which auxiliary lines you will draw and why they are helpful. For example:
   - "Connect point A to point B to form segment AB"
   - "Draw a perpendicular from point P to plane ABC, with foot H"
   - "Take the midpoint M of edge AB, connect M to C"
   - "Extend line DE to intersect plane ABC at point F"

3. After describing the auxiliary lines, provide a step-by-step solution using these auxiliary constructions.

4. Show all intermediate calculations and reasoning, including:
   - Distance calculations
   - Angle calculations
   - Volume/area calculations if needed

5. State the final answer clearly.

Please think step by step, starting with the auxiliary line construction.
```

### 4.2 Visual CoT

**Stage 1 Generation Prompt：**
```
You are given a solid geometry problem with a 3D diagram. Analyze the problem and create an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {question}

Instructions:
1. KEEP all original elements exactly as they are (all points, edges, faces, labels)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines). Common auxiliary constructions include:
   - Perpendiculars from a point to a plane or line
   - Line segments connecting specific points
   - Midpoints of edges with connections
   - Extended lines to find intersections
   - Parallel lines through specific points
   - Cross-sections of the solid
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original 3D figure with extra auxiliary lines drawn on top

Generate an enhanced 3D diagram that preserves the original and adds helpful auxiliary constructions.
```

**Stage 2 Question Prompt：**
```
You are given a solid geometry problem.

Problem: {question}

You are given TWO images:
1) ORIGINAL DIAGRAM: The 3D solid geometry figure as given
2) AUXILIARY DIAGRAM: The same figure with auxiliary constructions (extra lines) added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added (perpendiculars, connecting segments, midpoints, etc.)
2. Identify the geometric relationships established by these auxiliary lines
3. Use these constructions to set up your solution approach
4. Apply relevant theorems (Pythagorean theorem in 3D, properties of perpendiculars, volume formulas, etc.)
5. Show your step-by-step solution with clear calculations
6. State the final numerical answer

Solve this problem step by step using the auxiliary constructions.
```

---

## 5. IllusionBench

### 5.1 直接生成（带选项）

**Shape 任务：**
```
You are given an image where scene elements form an abstract SHAPE.
Task: Identify what shape is hidden in this image.

Options: [{shape_options}]

Reply with ONLY ONE word from the options above.
```

Shape 选项：
- ICON: animal, vehicle, stationary, sport, music, face_emoji
- LOGO: tesla, starbucks, mcdonalds, adidas, reebok, bmw, ubuntu, benz, telegram, nike, apple, puma, facebook, playstation, instagram, audi, olympics, google, spotify, amazon, nasa
- IN: guitar, teapot, cat, paper_clip, bird, dolphin, mug, bicycle, bottle, panda, dog, sailboat, car, fork, scooter, airplane

**Scene 任务：**
```
You are given an image depicting a SCENE.
Task: Identify the scene.

Reply in ONE line using this format:
Scene: <scene>
```

Scene 选项（通用）：Underwater_ruins, Time_square, Medieval_Village, City, Museum, Cloud, Ocean, Sand_dune, Bazaar_market, Forest, Origami

### 5.2 Visual CoT

**Stage 1 Generation Prompt：**

| 任务类型 | Generation Prompt |
|----------|-------------------|
| Shape | This image shows a scene where elements are carefully arranged to form a hidden shape. Your task: Extract and visualize this hidden shape. Generate a clear image that highlights the shape's outline, contours, and structure. Make the hidden shape prominent and easily recognizable. |
| Scene | This image depicts a specific scene or environment. Your task: Analyze and enhance the scene characteristics. Generate a clear visualization that emphasizes the environmental features and setting. |

**Stage 2 Question Prompt：**

Shape 任务：
```
You are given TWO images: the original image and an auxiliary visualization.
The image shows scene elements forming an abstract SHAPE.
Task: Identify what shape is hidden in this image.

Options: [{shape_options}]

Reply with ONLY ONE word from the options above.
```

Scene 任务：
```
You are given TWO images: the original image and an auxiliary visualization.
The image depicts a SCENE.
Task: Identify what scene is shown in this image.

Options: [{scene_options}]

Reply with ONLY ONE word from the options above.
```

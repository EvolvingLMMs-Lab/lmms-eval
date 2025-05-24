class Prompts:
    def __init__(self):
        self.event_system_prompt = "You are a video analysis expert specializing in evaluating the accuracy of video captions, particularly the descriptions of the events in a video. Please carefully analyze the user-provided caption and compare it to each provided event. Determine whether the caption contains the event."

        self.action_system_prompt = "You are a video analysis expert specializing in evaluating the accuracy of video captions, particularly the descriptions of actions in a video. Please carefully analyze the user-provided caption, compare it to the provided action and complete the task."

        self.object_category_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of objects in an image. Please carefully analyze the user-provided caption, compare it to the provided object and complete the task."

        self.object_color_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of the color of objects in an image. Please carefully analyze the user-provided caption, compare it to the provided object color and complete the task."

        self.object_number_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of the number of objects in an image. Please carefully analyze the user-provided caption, compare it to the provided object number and complete the task."

        self.dynamic_object_number_system_prompt = "You are a video analysis expert specializing in evaluating the accuracy of video captions, particularly the descriptions of the number of objects in a video. Please carefully analyze the user-provided caption, compare it to the provided object number and complete the task."

        self.spatial_relation_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of the spatial relationship between objects in an image. Please carefully analyze the user-provided caption, compare it to the provided spatial relationship between objects and complete the task."

        self.scene_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of the scene in an image. Please carefully analyze the user-provided caption, compare it to the provided scene and complete the task."

        self.camera_angle_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of camera angle in an image. Please carefully analyze the user-provided caption and complete the classification task."
        self.camera_angle_category_explains = [
            "level angle: Horizontal shooting of the subject (flat shot)",
            "high angle: Shooting from above the subject (overhead shot)",
            "low angle: Shooting from below the subject (upward shot)",
            "dutch angle: The lens has a certain angle of deflection along the central axis, making the horizon crooked",
        ]
        self.camera_angle_categories = [c.split(":")[0] for c in self.camera_angle_category_explains]

        self.camera_movement_system_prompt = "You are a video analysis expert specializing in evaluating the accuracy of video captions, particularly the descriptions of camera movements in the videos. Please carefully analyze the user-provided caption and complete the classification task."
        self.camera_movement_category_explains = [
            "left: the camera angle swings left (pan left), or the camera moves left (track left)",
            "right: the camera angle swings right (pan right), or the camera moves right (track right)",
            "up: the camera angle swings up (tilt up), or the camera moves up (boom up)",
            "down: the camera angle swings down (tilt down), or the camera moves down (boom down)",
            "in: camera pushes toward the subject (dolly in), or enlarges the frame (zoom in)",
            "out: camera moves away the subject (dolly out), or expands the visible area, makeing the subject appear smaller (zoom out)",
            "fixed: camera is almost fixed and does not change",
        ]
        self.camera_movement_categories = [c.split(":")[0] for c in self.camera_movement_category_explains]

        self.OCR_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of the OCR texts in an image. Please carefully analyze the user-provided caption, compare it to the provided OCR texts and complete the task."

        self.style_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of the image style. Please carefully analyze the user-provided caption and complete the classification task."
        self.style_category_explains = [
            "realistic: Represents subjects truthfully with lifelike detail and accuracy.",
            "animated: Created using 2D images or 3D computer-generated imagery (CGI), e.g., cartoon, anime",
            "special effect: Creates illusions through practical or digital techniques to enhance visuals.",
            "old-fashioned: Emulates historical aesthetics like vintage or classical artistic styles.",
            "pixel art: Retro digital art using blocky pixels for a nostalgic, low-res look.",
            "sketch art: Rough, expressive drawings emphasizing line work and spontaneity.",
            "abstract art: Non-representational art focused on shapes, colors, and emotions over realism.",
            "impressionism art: Captures fleeting light/moments with visible brushstrokes and vibrant color dabs.",
            "cubism art: Depicts subjects through fragmented geometric planes and multiple perspectives.",
        ]
        self.style_categories = [c.split(":")[0] for c in self.style_category_explains]

        self.character_identification_system_prompt = "You are an image analysis expert specializing in evaluating the accuracy of image captions, particularly the descriptions of person/character identification in an image. Please carefully analyze the user-provided caption, compare it to each provided name of the person/character and complete the task."

    def get_prompts_by_task(self, task, caption, anno):
        if task == "event":
            return self.get_event_prompts(caption, anno)
        if task == "action":
            return self.get_action_prompts(caption, anno)
        if task == "object_category":
            return self.get_object_category_prompts(caption, anno)
        elif task == "object_number":
            return self.get_object_number_prompts(caption, anno)
        elif task == "dynamic_object_number":
            return self.get_dynamic_object_number_prompts(caption, anno)
        elif task == "object_color":
            return self.get_object_color_prompts(caption, anno)
        elif task == "spatial_relation":
            return self.get_spatial_relation_prompts(caption, anno)
        elif task == "scene":
            return self.get_scene_prompts(caption, anno)
        elif task == "camera_angle":
            return self.get_camera_angle_prompts(caption)
        elif task == "camera_movement":
            return self.get_camera_movement_prompts(caption)
        elif task == "OCR":
            return self.get_OCR_prompts(caption, anno)
        elif task == "style":
            return self.get_style_prompts(caption)
        elif task == "character_identification":
            return self.get_character_identification_prompts(caption, anno)
        else:
            raise ValueError(f"Wrong task type: {task}")

    def get_event_prompts(self, caption, event):
        event_user_prompt = (
            "Given a video caption and an event as follows:\n"
            f"Video Caption: {caption}\n"
            f"Event: {event}\n"
            "Please analyze the video caption. Determine whether the provided event is described in the caption, and explain why. Note it can be considered mentioned as long as the caption contains an expression with a similar meaning to the event provided.\n"
            "Give score of 0 if the caption is totally irrelative to the provided event. Give score of 1 if the caption mentions the provided event correctly. Give score of -1 if the caption mentions the relative event give a wrong description.\n"
            "Output a JSON formed as:\n"
            '{"event": "copy provided event here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.event_system_prompt, event_user_prompt

    def get_action_prompts(self, caption, action):
        action_user_prompt = (
            "Given a video caption and an action as follows:\n"
            f"Video Caption: {caption}\n"
            f"Action: {action}\n"
            "Please analyze the video caption. Determine whether the provided action is mentioned in the caption, and explain why. Note it can be considered mentioned as long as the caption contains an expression with a similar meaning to the action provided.\n"
            "Give score of 0 if the caption does not mention ANY actions (including the provided action and any other action description). Give score of 1 if the caption mentions the provided action. Give score of -1 if the provided action is not mentioned in the caption.\n"
            "Output a JSON formed as:\n"
            '{"action": "copy provided action here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.action_system_prompt, action_user_prompt

    def get_object_category_prompts(self, caption, category):
        object_category_user_prompt = (
            "Given an image caption and an object as follows:\n"
            f"Image Caption: {caption}\n"
            f"Object: {category}\n"
            "Please analyze the image caption. Determine whether the provided object is mentioned in the caption, and explain why. Note it can be considered mentioned as long as the caption contains an expression with a similar meaning to the object provided.\n"
            "Give score of 0 if the caption does not mention ANY objects (including the provided object and any other objects). Give score of 1 if the caption mentions the provided object. Give score of -1 if the object is not mentioned in the caption.\n"
            "Output a JSON formed as:\n"
            '{"object_category": "copy provided object here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.object_category_system_prompt, object_category_user_prompt

    def get_object_number_prompts(self, caption, number):
        object_category, object_number = list(number.items())[0]
        object_number_user_prompt = (
            "Given an image caption and the number of an object with format {object: number} as follows:\n"
            f"Image Caption: {caption}\n"
            f"Object Number: {{{object_category}: {object_number}}}\n"
            f"Please analyze the image caption. Determine whether the provided object number is correctly described in the caption, and explain why. You may need to count in the caption to determine how many the provided objects it describes.\n"
            "Give score of 0 if the caption does not mention the specific number of provided object (including the use of words such as 'some' and 'various' in the caption rather than giving specific numbers) or not mention the provided object. Give score of 1 if the caption counts the provided object correctly. Give score only of -1 if the caption counts the wrong number of the provided object.\n"
            "Output a JSON formed as:\n"
            '{"object_number": "copy the provided {object: number} here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.object_number_system_prompt, object_number_user_prompt

    def get_dynamic_object_number_prompts(self, caption, number):
        dynamic_object_number_user_prompts = []
        for object_category, object_number in number.items():
            dynamic_object_number_user_prompt = (
                "Given a video caption and the number of an object with format {object: number} as follows:\n"
                f"Image Caption: {caption}\n"
                f"Object Number: {{{object_category}: {object_number}}}\n"
                f"Please analyze the video caption. Determine whether the provided object number is correctly described in the caption, and explain why. You may need to count in the caption to determine how many the provided objects it describes. Note you can never infer the number if the caption only gives 'some', 'several' without specific numbers.\n"
                "Give score of 0 if the caption does not mention the specific number of provided object (including the use of words such as 'some' and 'various' in the caption rather than giving specific numbers) or not mention the provided object. Give score of 1 if the caption counts the provided object correctly. Give score only of -1 if the caption counts the wrong number of the provided object.\n"
                "Output a JSON formed as:\n"
                '{"object_number": "copy the provided {object: number} here", "score": "put your score here",  "reason": "give your reason here"}\n'
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
            )
            dynamic_object_number_user_prompts.append(dynamic_object_number_user_prompt)
        return self.dynamic_object_number_system_prompt, dynamic_object_number_user_prompts

    def get_object_color_prompts(self, caption, color):
        object_category, object_color = list(color.items())[0]
        object_color_user_prompt = (
            "Given an image caption and the color of an object with format {object: color} as follows:\n"
            f"Image Caption: {caption}\n"
            f"Object Color: {{{object_category}: {object_color}}}\n"
            "Please analyze the image caption. Determine whether the provided object color is correctly described in the caption, and explain why.\n"
            "Give score of 0 for the following two situations:\n"
            "1) The provided object is not mentioned in the caption. Note it can be considered mentioned as long as the caption contains an expression with a similar meaning to the object provided.\n"
            "2) The caption does not mention the specific color of provided object\n"
            "Give score of 1 if the caption describes the object color correctly. Give score of -1 only if the caption gives the wrong color. Note it can be considered correct if the caption contains an expression with a similar meaning to the provided color.\n"
            "Output a JSON formed as:\n"
            '{"object_color": "copy the provided {object: color} here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.object_color_system_prompt, object_color_user_prompt

    def get_spatial_relation_prompts(self, caption, spatial_relation):
        spatial_relation_user_prompt = (
            "Given an image caption and the spatial relationship between two objects as follows:\n"
            f"Image Caption: {caption}\n"
            f"Spatial Relationship: {spatial_relation}\n"
            "Please analyze the image caption. Determine whether the provided spatial relationship is correctly decribed in caption, and explain why.\n"
            "Give score of 0 if the caption does not mention the spatial relationship between objects or not mention the objects. Give score of 1 if the caption describes the spatial relationship correctly. Give score of -1 only if the caption describes the wrong spatial relationship.\n"
            "Output a JSON formed as:\n"
            '{"spatial_relation": "copy the provided spatial relationship here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.spatial_relation_system_prompt, spatial_relation_user_prompt

    def get_scene_prompts(self, caption, scene):
        scene_user_prompt = (
            "Given an image caption and a scene as follows:\n"
            f"Image Caption: {caption}\n"
            f"Scene: {scene}\n"
            "Please analyze the image caption. Determine whether the provided scene is included in the caption, and explain why.\n"
            "Give score of 0 if the caption does not mention ANY scene information (including the provided scene and any other scenes). Give score of 1 if the caption mentions the provided scene. Give score of -1 only if the scene is not mentioned in the caption.\n"
            "Output a JSON formed as:\n"
            '{"scene": "copy the provided scene here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.scene_system_prompt, scene_user_prompt

    def get_camera_angle_prompts(self, caption):
        camera_angle_user_prompt = (
            "Given an image caption, your task is to determine which kind of camera angles is included in the caption.\n"
            f"Image Caption: {caption}\n"
            f"Please analyze the image caption and classify the descriptions of camera angles into the following categories: {self.camera_angle_categories}\n"
            "Here are the explanations of each category: " + "\n".join(self.camera_angle_category_explains) + "\n"
            "If the caption explicitly mentions one or some of the above camera angle categories, write the result of the categories with a python list format into the 'pred' value of the json string. You should only search the descriptions about the camera angle. If there is no description of the camera angle in the image caption or the description does not belong to any of the above categories, write 'N/A' into the 'pred' value of the json string.\n"
            "Output a JSON formed as:\n"
            '{"pred": "put your predicted category as a python list here", "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.camera_angle_system_prompt, camera_angle_user_prompt

    def get_camera_movement_prompts(self, caption):
        camera_movement_user_prompt = (
            "Given a video caption, your task is to determine which kind of camera movement is included in the caption.\n"
            f"Video Caption: {caption}\n"
            f"Please analyze the video caption and classify the descriptions of camera movement into the following categories: {self.camera_movement_categories}\n"
            f"Here are the explanations of each category: " + "\n".join(self.camera_movement_category_explains) + "\n"
            "If the caption explicitly mentions one or some of the above camera movement categories, write the result of the categories with a python list format into the 'pred' value of the json string. Note do not infer the camera movement categories from the whole caption. You should only search the descriptions about the camera movement. If there is no description of the camera movement in the video caption or the description does not belong to any of the above categories, write 'N/A' into the 'pred' value of the json string.\n"
            "Output a JSON formed as:\n"
            '{"pred": "put your predicted category as a python list here", "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.camera_movement_system_prompt, camera_movement_user_prompt

    def get_OCR_prompts(self, caption, OCR_text):
        OCR_user_prompt = (
            "Given an image caption and an OCR text as follows:\n"
            f"Image Caption: {caption}\n"
            f"OCR Text: {OCR_text}\n"
            f"Please analyze the image caption. Determine whether the provided text is described correctly in the caption, and explain why.\n"
            "Give score of 0 if there is no description about the provided OCR text in the caption. Give score of 1 if the caption refers the text and recognizes correctly. Give score of -1 if the recognization result is wrong in the caption.\n"
            "Output a JSON formed as:\n"
            '{"OCR": "copy the provided real OCR text here", "score": put your score here, "reason": "give your reason here"},\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.OCR_system_prompt, OCR_user_prompt

    def get_style_prompts(self, caption):
        style_user_prompt = (
            "Given an image caption, your task is to determine which category of image style is included in the caption.\n"
            f"Image Caption: {caption}\n"
            f"Please analyze the image caption and classify the descriptions of the image style into the following categories: {self.style_categories}\n"
            f"Here are the explanations of each category: " + "\n".join(self.style_category_explains) + "\n"
            "If the description of the image style belongs to one or some of the above categories, write the result of the categories with a python list format into the 'pred' value of the json string. Focus more on the artistic style part in the caption. If there is no description of the image style in the image caption or the description does not belong to any of the above categories, write 'N/A' into the 'pred' value of the json string.\n"
            "Output a JSON formed as:\n"
            '{"pred": "put your predicted category as a python list here", "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.style_system_prompt, style_user_prompt

    def get_character_identification_prompts(self, caption, character_identification):
        character_identification_user_prompt = (
            "Given an image caption and the name of a person/character as follows:\n"
            f"Image Caption: {caption}\n"
            f"name: {character_identification}\n"
            "Please analyze the image caption. Determine whether the provided name of person/character is included in the caption, and explain why.\n"
            "Give score of 0 if the caption does not mention any names. Give score of 1 if the caption mentions the provided name correctly. Give score of -1 if the name in the caption gives a wrong name.\n"
            "Output a JSON formed as:\n"
            '{"character_identification": "copy the provided name here", "score": "put your score here",  "reason": "give your reason here"}\n'
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Do not add Markdown syntax. Output:"
        )
        return self.character_identification_system_prompt, character_identification_user_prompt

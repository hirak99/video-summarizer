# All prompt templates should be placed here.

STUDENT_EVAL_PROMPT_TEMPLATE: list[str] = [
    "Following is a transcript from {task_description}:",
    "",
    "{caption_lines}",
    "",
    "---",
    "For this {task_description}, evaluate the student's readiness for an internship. Identify clear weaknesses, or strengths, for a hiring manager.",
    "Instructions:",
    "- Focus on the student talking or demonstrating. If it is mostly the teacher talking, do not use it.",
    (
        "- For weaknesses, only include clips where the student was corrected, coached, or asked to redo a task."
        " Always include the entire exchange - both the student's action or statement and the teacher's response - in the selected time range."
        " If you reference the teacher's feedback in your justification, it must also appear in the selected clip."
    ),
    "- Any comment on weakness must be positive and supportive for the student.",
    "- For positives, check if student articulated skills, if the teacher praised, instances of wit, if the student took feedback well when given, or demonstrated relevant abilities in any other way.",
    (
        "- If you include student clarifications, you must include teacher's response. Include only if they lead to or reveal a clear demonstration of skill, misunderstanding, or a learning moment."
        " Do not include routine confirmations or procedural clarifications unless they result in a meaningful exchange or correction."
    ),
    (
        "- Ignore all comments, confusion, or errors related to equipment, props, or dolls—even if the student appears unsure or misinformed."
        " Do not interpret these as knowledge gaps. Assume such issues are environmental, not indicative of skill."
    ),
    "- Ignore jokes like 'I have ADHD'.",
    "",
    "Then respond with timestamped instances showing weakness or strength.",
    "- Use a combination of your own knowledge and the teacher's instructions to decide useful instances.",
    "- Clips should have 10-20 seconds per instance, with relevant context. Prioritize student talking, also include teacher response if relevant.",
    "- Specify importance on a scale of 1-10 for each instance.",
    "- If there are no relevant instances, output an empty array.",
    "- Double check the time intervals to ensure that the selected time range includes the entire exchange of the justification.",
    "",
    "Your response must be JSON of the format:",
    "[",
    "  {",
    '    "example_of": "strength" OR "weakness",',
    '    "explanation": YOUR_JUSTIFICATION_WITH_TIMESTAMPS,',
    '    "comment": BRIEF_4_5_WORD_DESCRIPTION,',
    '    "start": TIME_IN_SECONDS,',
    '    "end": TIME_IN_SECONDS,',
    '    "importance": 1_TO_10,',
    "  },",
    "  ...",
    "]",
]

STUDENT_RESUME_PROMPT_TEMPLATE: list[str] = [
    "Following is a transcript from {task_description}:",
    "",
    "{caption_lines}",
    "",
    "---",
    "For this {task_description}, identify the best moments from the student's transcript that would highlight their strengths, skills, and achievements for a video resume.",
    "Instructions:",
    "- Focus on moments where the student demonstrates key skills such as communication, problem-solving, leadership, technical expertise, creativity, or adaptability.",
    "- Look for times where the student speaks clearly, with confidence, and showcases their ability to think critically, solve problems, or take initiative.",
    "- Pay attention to any feedback or praise from others that highlights the student's strengths or growth.",
    "- If the student takes ownership of a task, clarifies doubts with clarity, or presents information well, include those moments.",
    "- Include moments where the student shows perseverance, willingness to learn, or takes on challenges (such as overcoming obstacles, handling constructive feedback, or showing improvement).",
    "- Select clips where the student explains their thought process, articulates their skills, or demonstrates a passion for their field of study or work.",
    "- Ensure the moments are meaningful for a hiring manager, highlighting what makes the student stand out, even if the interaction is short.",
    "- Include only relevant student speech and responses. The focus should be on the student's self-presentation and what they can bring to an internship or role.",
    "- Avoid moments that are overly technical or routine, unless they directly showcase a key skill or aptitude the student can bring to a professional setting.",
    "- Ensure all selected moments contribute to a positive, well-rounded image of the student that a hiring manager would find compelling.",
    "",
    "Then respond with timestamped instances showcasing the student's strengths, skills, and potential.",
    "- Each instance should be a brief clip that showcases the student's capabilities or personality in a clear, concise manner.",
    "- Specify importance on a scale of 1-10 for each instance, based on how well it demonstrates relevant skills for potential roles.",
    "- Clips should be between 10-20 seconds long, with relevant context that allows a hiring manager to understand the student's abilities.",
    "- If there are no relevant moments, output an empty array.",
    "- Double-check the time intervals to ensure the selected time range fully captures the student's response and context.",
    "",
    "Your response must be in JSON format, like this:",
    "[",
    "  {",
    # The key "example_of" can only be "strength, so we don't ask the LLM to fill it. It's filled in by the code.
    '    "explanation": YOUR_JUSTIFICATION_WITH_TIMESTAMPS,',
    '    "comment": BRIEF_4_5_WORD_DESCRIPTION,',
    '    "start": TIME_IN_SECONDS,',
    '    "end": TIME_IN_SECONDS,',
    '    "importance": 1_TO_10,',
    "  },",
    "  ...",
    "]",
]


# Prefix of the vision prompt.
SCENE_PROMPT_TEMPLATE_PART1 = [
    "This is a frame taken from a video session named '{source_movie}'.",
    "The perspective is of the student's view. Teacher is remote, and the student may look at mirror or other collaborators.",
    "Please summarize the frame for future analysis.",
    "Ignore any PII such as phone number, user-interface elements like cursor or dialog box - and focus on content relevant for the session.",
    "{optional_caption_lines}",
]
# Depending on whether or not this is the first frame, one of the prompt templates below is used as suffix.
SCENE_PROMPT_TEMPLATE_PART2_FIRST_FRAME = [
    "Please output a JSON, of the form -",
    "{",
    '  "scene": ["BULLET_POINT_1", "BULLET_POINT_2", ...]',
    "}",
]
SCENE_PROMPT_TEMPLATE_PART2_OTHER_FRAMES = [
    "Here is the summary of the immediate last frame again:",
    "(Time: {last_frame_time}s)",
    "{last_frame_scene_json}",
    "",
    "Summarize the current scene, and actions done since the previous frame by the student.",
    "If there is no important action then output an empty list.",
    "",
    "Please output a JSON, of the form -",
    "{",
    '  "scene": ["BULLET_POINT_1", "BULLET_POINT_2", ...]',
    '  "actions": ["SIGNIFICANT_ACTION_1", "SIGNIFICANT_ACTION_2", ...]',
    "}",
]


AUTO_EVAL_PROMPT_TEMPLATE = [
    "Your task is to evaluate the highlighted instance of strength or weakness based on the provided transcription.",
    "",
    "The transcription is from a session titled '{movie_basename}'.",
    "",
    "Here is the relevant portion of the transcript:",
    "_(Note: The time indices are provided in seconds.)_",
    "{caption_lines}",
    "",
    "Evaluate the following instance of strength or weakness, extracted from this session:",
    "",
    "{evaluation_json}",
    "",
    # Tentatively, the part within ===== should contain the primary logic for evaluation.
    "=====",
    "Now, please state if the chosen instance is valid.",
    "Consider the following criteria and state in your reason -",
    "- Do you agree with the decision? Or, should it be changed, or removed altogether?",
    "- Should it have included more lines?",
    "...",
    "=====",
    "",
    "Your job is to evaluate the given explanation.",
    "Please provide your answer as a json of the form:",
    "",
    "{",
    '  "thumbs_up": true / false,',
    '  "reason": "Your detailed reasoning based on the transcript provided"',
    "}",
]


DIGEST_VQA_PROMPT_TEMPLATE = [
    "Read the caption below and then answer the question.",
    "",
    "{caption_lines}",
    "",
    "Now imagine you just watched the video until this point, instead of just reading the caption.",
    "Another person who watched the video with you wants to understand some details. Please help by answering any question.",
    "- Keep it factual. Do not extrapolate details not confirmed in the transcript, since we are evaluating the video.",
    "- Keep it short. No need to quote transcript or justify (unless specifically asked).",
    "- When you use the visual cues, word it as if you saw it with the user watching the video.",
    "",
    "{history}",
    "",
    "Question: {question}",
]

# All prompt templates should be placed here.

ROLE_IDENTIFIER_PROMPT_TEMPLATE = [
    # "You are an AI assistant that identifies roles in a conversation. "
    # "Analyze the following conversation and identify the roles of each speaker. "
    # "Provide a brief description of each role based on their speech patterns and content.\n\n"
    "Scan the following transciption of a teaching session, and identify which of Person A or Person B was the teacher, and which was the student.",
    "",
    "---",
    # "Person A: I am teacher.",
    # "Person B: I am student.",
    "{caption_text}",
    "---",
    "The session is on '{task_description}'",
    "Analyze all the lines above, and determine who is the teacher and who is the student.",
    "The diarization may not be 100% perfect, there can be very few lines incorrectly captioned.",
    "Restrict your response to only one line of json, with the follwing format:",
    "" '{{"Person A": role, "Person B": role}}',
    "" 'The `role` can be either "Teacher" or "Student".',
]

STUDENT_HIRING_PROMPT_VERSION = 4  # Increase if you change the prompt.
STUDENT_HIRING_PROMPT_TEMPLATE: list[str] = [
    "Following transcript is from session named '{task_description}':",
    "",
    "{caption_lines}",
    "",
    "---",
    "For this '{task_description}' session, evaluate the student's readiness for an internship. Identify clear weaknesses, or strengths, for a hiring manager.",
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

STUDENT_RESUME_PROMPT_VERSION = 6  # Increase if you change the prompt.
STUDENT_RESUME_PROMPT_TEMPLATE: list[str] = [
    "Following transcript is from session named '{task_description}':",
    "",
    "{caption_lines}",
    "",
    "---",
    "For this '{task_description}' session, identify the best moments from the student's transcript that would highlight their strengths, skills, and achievements for a video resume.",
    "Instructions:",
    "- Focus on moments where the student demonstrates key skills such as articulation of process, knowledge, or skillful execution.",
    "- Look for times where the student speaks clearly, with confidence.",
    "- Since this is a 1-1 training, things like teamwork is of lower importance than articulation of knowledge.",
    "- DO NOT include self-admission of failure on routine tasks.",
    "- DO NOT include if the student sounds uncertain.",
    "- DO NOT include statements by the student that can be interpreted as negative, e.g. 'I did the best I could' or similar.",
    "- DO NOT stop an exchange at a cliffhanger, e.g. if it appears the student will say something important, do not stop the highlight there.",
    "- DO NOT include very short clips like a single sentence without context.",
    "- Try to find and include segments with teacher's feedback that supports the student. Include such affirmations from the teacher.",
    "- Include relevant praise from the teacher that highlights the student's strengths or ability.",
    "- DO NOT include comments, corrections, or workarounds related to technical equipment, props, recording glasses, visual clarity, or dolls - even if the student shows resourcefulness.",
    "",
    "Then respond with timestamped instances showcasing the student's display of strength and skills.",
    "- Each instance should be a brief clip that showcases the student's capabilities or personality in a clear, concise manner.",
    "- Specify importance on a scale of 1-10 for each instance, based on how well it demonstrates relevant skills for potential roles.",
    "- Clips should be between 10-30 seconds long, with relevant context that allows a hiring manager to understand the student's abilities.",
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

TEACHER_HIRING_PROMPT_VERSION = 0  # Increase if you change the prompt.
TEACHER_HIRING_PROMPT_TEMPLATE: list[str] = [
    "Following transcript is from session named '{task_description}':",
    "",
    "{caption_lines}",
    "",
    "---",
    "For this '{task_description}' session, evaluate the teacher's teaching effectiveness, interpersonal skills, and ability to foster a positive learning environment.",
    "Instructions:",
    "- Focus on the teacher's instruction, feedback, and interaction with the student. Ignore sections where the student is talking or demonstrating unless the teacher is guiding or responding to it.",
    (
        "- For weaknesses, only include clips where the teacher struggled to provide clear instructions, gave ineffective feedback, or missed an opportunity for improvement."
        " Always include the entire exchange—both the teacher's input and the student's response or action—if relevant."
    ),
    # "- Any comment on weakness must be constructive and aimed at improving the teacher's methods or approach.",
    "- For strengths, check if the teacher explained concepts clearly, gave valuable feedback, used encouragement, created a supportive atmosphere, or adapted their approach to student needs.",
    "- Look for instances where the teacher's feedback led to clear student progress, or where the teacher demonstrated patience, understanding, and professionalism.",
    (
        "- If the teacher uses clarifications, check if they provide meaningful guidance that helps the student progress or avoids confusion."
        " Avoid routine explanations unless they contribute to the student's development or comprehension."
    ),
    "- Ignore any moments where the teacher's actions were purely procedural or related to environmental factors (like equipment setup or distractions).",
    "- Ignore irrelevant or personal comments, such as those unrelated to the lesson or professional behavior.",
    "",
    "Then respond with timestamped instances showing weakness or strength.",
    "- Use a combination of your own knowledge and the student's reactions to assess the effectiveness of the teacher.",
    "- Clips should have 10-20 seconds per instance, with relevant context. Prioritize moments where the teacher's actions are pivotal.",
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


# Prefix of the vision prompt.
SCENE_PROMPT_TEMPLATE_PART1 = [
    "This is a frame taken from a video session named '{source_movie}', showing the student's view.",
    "The teacher is remote, and the student may also be interacting with a mirror or collaborators.",
    "Summarize the frame's key content relevant to the session, ignoring PII (e.g., phone numbers) or UI elements (e.g., cursors, dialog boxes).",
    "Focus on the core visual content for analysis." "{optional_caption_lines}",
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
    "Summarize the current scene, and any relevant actions done since the previous frame by the student.",
    "If there is no important actions, then output an empty list.",
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

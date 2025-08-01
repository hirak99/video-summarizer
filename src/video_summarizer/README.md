# Video Understanding & Summarization

This is a set of pipelines with which we create pipelines to understand, curate, and summarize a group of videos into a single 5 minutes highlight reel.

Specific application demonstrated here assumes the videos are of teacher-student sessions, with multiple teachers and students. The summary generates one file for each student, across all the sessions he or she participated in.

This uses the [flow](../flow/README.md) architecture.

In particulare we use two different flows here.

## Video Flow
A flow that is run to process each video.
```mermaid
flowchart TD;
    A[Video Input]
    A -- video --> M[PII Detection]
    A --> I
    C --> I[Visualize for Auditing]
    D --> I
    F --> I
    A -- audio --> C["(pyannote Local)<br>"Speaker Diarization]
    A -- audio --> B["(Whisper Local)<br>"Captioning]
    C --> D
    B --> D[Speaker Assignment]
    D --> F["(OpenAI o4-mini)<br>"Speaker Role Identification];
    F --> K[Role Aware Captions]
    D --> K
    K --> L["(OpenAI o4-mini)<br>"Student Evaluation]
    A -- video --> N["(OpenAI gpt-4.1)<br>Scene Understanding"]
    A --> LABELS["Manual Labeling"] --> N
    K --> N
    N --> L
    LABELS@{ shape: lean-r}
```

A brief explanation of some of the pipeline nodes is given below:


| Nodes                       | Explanation                                                                                                                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Captioning                  | Captioning of speech in videos. Captioning does not identify speakers.                                                                                                                                |
| Speaker Diarization         | Identifies *when* different speakers are talking, outputting time segments for each speaker. Done via pyannote models for voice activity detection, segmentation, and clustering of voice embeddings. |
| Speaker Assignment          | Using the diarization output, assigns speakers labels (e.g. SPEAKER_00, SPEAKER_01)  to the corresponding parts of the transcription.                                                                 |
| Speaker Role Identification | Associate speaker ids with roles given the context of the video.                                                                                                                                      |
| Summarization               | Summarize based on the audio given speaker roles and transcriptions.                                                                                                                                  |
| Visualization and Iteration | Visualization of the speaker roles is done as an overlay on the video.                                                                                                                                |
| Manual Labeling             | Done using https://github.com/hirak99/video-annotator to identify regions-of-interest in the vide.                                                                                                    |

## Student Flow
Once all the videos are processed, the following is run to summarize across videos.

```mermaid
flowchart TD
    X[Original Videos] --> B
    Y[Student Evaluations] --> A
    Z[Object Detections] -- blur --> B
    A[Hiring Highlight Curation] --> B[Compiled Movie]
    A --> C[Auto Eval Templates]
    X@{ shape: procs}
    Y@{ shape: procs}
    Z@{ shape: procs}
```

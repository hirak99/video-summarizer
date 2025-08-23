# Video Understanding & Summarization

This is a set of pipelines with which we create pipelines to understand, curate, and summarize a group of videos into a single 5 minutes highlight reel.

Specific application demonstrated here assumes the videos are of teacher-student sessions, with multiple teachers and students. The summary generates one file for each student, across all the sessions he or she participated in.

This uses the [flow](../flow/README.md) architecture.

In particular we use two different flows here.

## Video Flow
A flow that is run to process each video.
```mermaid
flowchart TD;
    A[Video Input]
    A -- video --> M[PII Detection]
    A --> I
    Diarization --> I[Visualize for Auditing]
    D --> I
    F --> I
    A -- audio --> Diarization["(pyannote Local)<br>"Speaker Diarization]
    A -- audio --> Captioning["(Whisper Local)<br>"Captioning]
    Diarization --> D
    Diarization --> RefineCaptions
    Captioning --> RefineCaptions[Refine Captions]
    RefineCaptions --> D[Speaker Assignment]
    D --> F["(OpenAI o4-mini)<br>"Speaker Role Identification];
    F --> K[Role Aware Captions]
    D --> K
    K --> HighlightSel["(OpenAI o4-mini)<br>"Highlight Selection]
    A -- video --> N["(OpenAI gpt-4.1)<br>Scene Understanding"]
    A --> LABELS["Manual Labeling"] --> N
    K --> N
    N --> HighlightSel
    LABELS@{ shape: lean-r}
    HighlightSel@{ shape: procs}
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
    Y[Selected Highlights] --> A
    Z[PII Detections] -- blur --> B
    A[Highlight Curation] --> B[Compiled Movie]
    A --> C[Auto Eval Templates]
    X@{ shape: procs}
    Y@{ shape: procs}
    Z@{ shape: procs}
```

## Video Question & Answering

Here's how VQA fits into the rest of the architecture -

```mermaid
flowchart LR
    A["Video Understanding"] --> B["Applications"]
    B ~~~ APP_LIST["Summary Reels,<br>Issue Detection,<br>etc."]
    A -. implements .-> C["VQA"]
    C --> D["User Interface"] & E["Automated Evaluation"]
    E -. measures .-> A

    APP_LIST@{ shape: comment}
```

- VQA uses an abstraction ([ref](./vqa/abstract_vqa.py)) that is independent of the rest of video understanding implementation. This removes mental barriers in evolving, or even replacing, the underlying system without being encumbered by VQA's implementation.

- While the abstraction is independent, VQA's implementation will be tied to specific video-understanding systems.

### Auto Eval

One primary usa case of VQA is to facilitate the automatic evaluation of our system.

This is achieved by setting up questions and precise answers at various points in test videos, which VQA can then process to come up with its own answers, and can be scored automatically.

### Demo

Below is a demo of Video Question Answering on an EKG training video.

![](https://github.com/hirak99/_media_assets/blob/master/vqa_demo_20250803_whiteout.gif?raw=true)

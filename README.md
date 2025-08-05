# Video Understanding

## Project Overview

This repository hosts two closely related projects designed to build a scalable video-understanding and summarization pipeline.

1. **[Flow](./src/flow/README.md)**

   A DAG-based workflow manager that offers task resumption, caching, logging, error handling, and persistence, ensuring efficient and fault-tolerant execution of machine learning pipelines.

2. **[Video Understanding](./src/video_understanding/README.md)**

   A video understanding, summarization, and question-answering system that automatically condenses hours of teacher-student sessions into a single concise 5-minute curated highlight.

### Related Project

- **[Video Annotator](https://github.com/hirak99/video-annotator)**

  It is a web-based tool for manual video annotation, to annotate lattice-aligned rectangular ROIs in the videos. It was primarily made for this project to quickly annotate map the teacher / student areas. It is also used to mark simple regions for adhoc blurring.

## Real-Life Usage

This project originally began as a [private project](https://github.com/hirak99/process_graph) (restricted access).

After extracting the domain-specific data, the first public release was made with over 280 commits squashed into a single release.

The parent project now includes this repository as a submodule, while overriding the domain-specific data.

## Setup

Please see [SETUP.md](./SETUP.md) for setup instructions.

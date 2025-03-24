KubeFlow Pipeline + GitHub Actions and GCP 🚀

This is a simple example of setting up a Kubeflow Pipeline and automating deployment with GitHub Actions using GCP services. 

Here’s a quick rundown of how it works:

Pipeline Setup: The pipeline is written in Python using the Kubeflow Pipelines framework.

Execution: The pipeline runs on Vertex AI.

Deployment: A GCP Cloud Function is used to kick off the pipeline execution.

Scheduling: A GCP Scheduler is set up to handle the pipeline execution timing. At the scheduled time, it sends a message to a Pub/Sub topic, which kicks off the process.

Trigger: The Pub/Sub topic activates the Cloud Function, starting the pipeline on Vertex AI.

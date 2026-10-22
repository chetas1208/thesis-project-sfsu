#!/usr/bin/env python3
"""Build the LARGE SYNTHETIC course catalogue used by the recommender benchmark.

Output: inputs/courses/course_catalog_large.jsonl  (same schema as course_catalog.jsonl)

PROVENANCE / HONESTY NOTES
--------------------------
* No real course catalogue (Coursera / Udemy / edX / university) exists in this repository or in the
  sibling 'Resume+Job Data' folder (that folder is empty).  Every course below is therefore SYNTHETIC:
  authored by hand as (title, level, hours, taught skills, prerequisites) rows and expanded with
  skill-level phrase templates into learning outcomes / descriptions.
* Providers are fictional ("Synthetic <Area> Academy"); no URLs, prices or ratings are claimed to be
  real.  Ratings / costs are deterministic pseudo-random placeholders (hash of course_id).
* What IS grounded in real data: the skill vocabulary.  The taught-skill vocabulary was checked against
  (a) the project's skill alias taxonomy (data/skillllm_v2/taxonomy/skill_aliases.jsonl) and
  (b) document frequencies over the 1,887 real job postings in
      SkillLLM_v2/Extract_jobDesc/data/job_descriptions.jsonl (optional, pass --postings; if present the
      per-skill posting count is stored in metadata.real_posting_df and a coverage statistic is printed).
  Course structure (levels, prerequisite chains, duration) reflects common curriculum design and is
  NOT derived from any real provider.

Run:  PYTHONPATH=src python scripts/build_course_catalog_large.py [--postings PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Skill phrases: one outcome-style clause per skill (used for outcomes + descriptions).
# --------------------------------------------------------------------------------------
PH = {
    # programming / software engineering
    "Python": "write clean, idiomatic Python programs using functions, modules and the standard library",
    "Java": "build object-oriented Java applications with collections, generics and exceptions",
    "JavaScript": "write modern ES6+ JavaScript including closures, promises and async/await",
    "TypeScript": "add static types to JavaScript codebases using TypeScript interfaces and generics",
    "Go": "implement concurrent network services in Go with goroutines and channels",
    "C++": "write memory-aware C++ using RAII, templates and the STL",
    "Rust": "write memory-safe systems code using Rust ownership and lifetimes",
    "Bash Scripting": "automate routine tasks with Bash scripts, pipes and cron",
    "Object-Oriented Programming": "model problem domains with classes, inheritance and polymorphism",
    "Data Structures": "implement lists, trees, graphs, heaps and hash tables and reason about their trade-offs",
    "Algorithms": "design and analyse sorting, searching and graph algorithms with Big-O complexity",
    "Problem Solving": "decompose ambiguous problems and solve them systematically under time pressure",
    "Functional Programming": "apply immutability, higher-order functions and pure functions",
    "Concurrency": "reason about threads, locks and race conditions in concurrent programs",
    "Software Testing": "plan test cases, apply equivalence partitioning and report defects",
    "Test Automation": "automate regression suites and integrate them into build pipelines",
    "Unit Testing": "write isolated unit tests with fixtures, mocks and coverage reports",
    "Test-Driven Development": "drive designs with a red-green-refactor test-first workflow",
    "Design Patterns": "apply creational, structural and behavioural design patterns",
    "Code Review": "review pull requests for correctness, readability and maintainability",
    "Debugging": "locate and fix defects using debuggers, logging and bisection",
    "Performance Profiling": "profile CPU and memory hot spots and remove bottlenecks",
    "Git": "manage source code with branches, merges, rebases and pull requests",
    "Software Architecture": "evaluate architectural styles and document design decisions",
    "System Design": "design scalable distributed systems including caching, queues and load balancing",
    "Linux": "work productively on the Linux command line, permissions and processes",
    "Linux System Administration": "administer Linux servers: users, services, storage and kernel tuning",
    "Computer Networking": "explain the OSI model, routing, DNS and HTTP",
    "TCP/IP": "analyse TCP/IP traffic and troubleshoot connectivity",
    "Dynamic Programming": "solve optimisation problems with memoisation and dynamic programming",
    "Requirements Engineering": "elicit, specify and validate software requirements and user stories",
    # data
    "SQL": "write SQL queries with joins, aggregations and subqueries",
    "PostgreSQL": "design and operate PostgreSQL databases with indexes and transactions",
    "MySQL": "configure, back up and tune MySQL servers",
    "Database Design": "design normalised relational schemas with keys and constraints",
    "Database Administration": "manage replication, backups and access control for relational databases",
    "Query Optimization": "read execution plans and tune slow SQL queries",
    "NoSQL": "choose and model data for document, key-value and column stores",
    "MongoDB": "model and query document data in MongoDB with aggregation pipelines",
    "Redis": "use Redis data structures for caching, queues and rate limiting",
    "Caching": "apply cache-aside, write-through and invalidation strategies",
    "Data Modeling": "build conceptual, logical and dimensional data models",
    "Data Warehousing": "design star schemas and load analytical warehouses",
    "Snowflake": "load, query and secure data in the Snowflake cloud warehouse",
    "ETL": "build extract-transform-load jobs with validation and incremental loads",
    "Data Pipelines": "design reliable batch data pipelines with scheduling, retries and monitoring",
    "Apache Airflow": "orchestrate dependent tasks as Airflow DAGs with sensors and operators",
    "dbt": "transform warehouse data with dbt models, tests and documentation",
    "Apache Spark": "process large datasets with Spark DataFrames and distributed execution",
    "Apache Kafka": "produce and consume event streams with Kafka topics and consumer groups",
    "Stream Processing": "implement windowed and stateful stream processing",
    "Hadoop": "explain HDFS and MapReduce and run jobs on a Hadoop cluster",
    "Big Data": "reason about partitioning, shuffling and scale-out data processing",
    "Pandas": "clean, reshape and aggregate tabular data with Pandas",
    "NumPy": "vectorise numerical computation with NumPy arrays",
    "Data Cleaning": "detect and repair missing, duplicate and inconsistent records",
    "Matplotlib": "produce publication-quality plots with Matplotlib",
    "Data Visualization": "choose effective charts and build clear visual explanations of data",
    "Tableau": "build interactive Tableau dashboards and calculated fields",
    "Power BI": "model data and publish interactive Power BI reports",
    "Business Intelligence": "define KPIs and deliver self-service reporting",
    "Excel": "analyse data in Excel with pivot tables, lookups and formulas",
    "Data Analysis": "answer business questions by querying, summarising and interpreting data",
    "Exploratory Data Analysis": "explore distributions, outliers and relationships before modelling",
    "Statistics": "apply descriptive statistics, sampling and confidence intervals",
    "Probability": "use probability distributions, conditional probability and Bayes' rule",
    "Hypothesis Testing": "design and interpret t-tests, chi-square tests and p-values",
    "A/B Testing": "design, power and analyse online controlled experiments",
    "R": "analyse and plot data in R with the tidyverse",
    "Time Series Analysis": "model trend and seasonality and produce forecasts",
    "Data Governance": "define ownership, lineage and access policies for enterprise data",
    "Data Quality": "implement data validation rules and quality monitoring",
    "Data Storytelling": "turn analysis into a narrative that drives decisions",
    # ML / AI
    "Linear Algebra": "compute with vectors, matrices, eigendecompositions and SVD",
    "Calculus": "use derivatives and gradients for optimisation",
    "Optimization": "formulate and solve convex and constrained optimisation problems",
    "Machine Learning": "train, tune and validate supervised machine learning models",
    "Scikit-Learn": "build model pipelines with scikit-learn",
    "Model Evaluation": "choose metrics, cross-validation schemes and diagnose over-fitting",
    "Feature Engineering": "construct, encode and select informative features",
    "Ensemble Methods": "combine models with bagging, boosting and stacking",
    "Gradient Boosting": "tune gradient-boosted trees such as XGBoost and LightGBM",
    "Unsupervised Learning": "apply clustering and dimensionality reduction to unlabelled data",
    "Anomaly Detection": "detect rare events with statistical and learned detectors",
    "Deep Learning": "train deep neural networks with backpropagation and regularisation",
    "Neural Networks": "implement feed-forward, convolutional and recurrent networks",
    "PyTorch": "build and train models with PyTorch tensors, autograd and data loaders",
    "TensorFlow": "build and train models with TensorFlow and Keras",
    "Computer Vision": "train convolutional models for image classification and detection",
    "Natural Language Processing": "tokenise, embed and classify text with NLP pipelines",
    "Transformers": "implement self-attention and transformer encoder-decoder architectures",
    "Large Language Models": "use and evaluate large language models via APIs and open weights",
    "Prompt Engineering": "design, test and version prompts for LLM applications",
    "Generative AI": "explain generative model capabilities, limits and business use cases",
    "Fine-Tuning": "fine-tune pretrained models with LoRA and supervised instruction data",
    "Retrieval-Augmented Generation": "build RAG systems with chunking, retrieval and grounded answers",
    "Vector Databases": "index and query embeddings in a vector database",
    "Recommender Systems": "build collaborative-filtering and content-based recommenders",
    "Reinforcement Learning": "train agents with Q-learning and policy gradients",
    "MLOps": "automate training, testing and release of machine learning systems",
    "Model Deployment": "package and serve models behind REST endpoints",
    "Model Monitoring": "monitor drift, latency and data quality of models in production",
    "Experiment Tracking": "track experiments, parameters and artefacts reproducibly",
    "Inference Optimization": "reduce model latency with quantisation, batching and compilation",
    "Responsible AI": "assess bias, fairness and explainability of models",
    "Data Science": "run an end-to-end data science project from question to communicated result",
    "Artificial Intelligence": "explain core AI concepts and where they apply",
    # cloud
    "Cloud Computing": "explain IaaS, PaaS and SaaS and the shared-responsibility model",
    "Amazon Web Services": "provision and operate core AWS services such as EC2, S3, VPC and IAM",
    "Microsoft Azure": "provision and operate core Azure services and resource groups",
    "Google Cloud Platform": "provision and operate core Google Cloud services",
    "Cloud Architecture": "design well-architected, resilient multi-tier cloud solutions",
    "Serverless": "build event-driven serverless applications",
    "AWS Lambda": "deploy and tune AWS Lambda functions with API Gateway and event triggers",
    "Infrastructure as Code": "define repeatable infrastructure as version-controlled code",
    "Terraform": "write Terraform modules, state and workspaces to provision cloud infrastructure",
    "Ansible": "automate server configuration with Ansible playbooks and roles",
    "Configuration Management": "keep fleets of servers consistently configured",
    "Cloud Networking": "design VPCs, subnets, routing and private connectivity in the cloud",
    "Cloud Cost Optimization": "analyse cloud bills and apply rightsizing and commitment discounts",
    "High Availability": "design multi-zone and multi-region failover",
    "Cloud Security": "secure cloud workloads, storage and networks",
    "Identity and Access Management": "design least-privilege roles, policies and federation",
    # devops
    "Docker": "build, tag and run container images with Dockerfiles and Compose",
    "Containerization": "package applications and dependencies into portable containers",
    "Kubernetes": "deploy and operate workloads on Kubernetes with pods, services and deployments",
    "Container Orchestration": "schedule, scale and heal containerised services",
    "Helm": "package and release Kubernetes applications with Helm charts",
    "CI/CD": "design automated build, test and deployment pipelines",
    "GitHub Actions": "author GitHub Actions workflows with matrices, caching and secrets",
    "Jenkins": "build declarative Jenkins pipelines and agents",
    "GitOps": "reconcile cluster state from Git using declarative delivery",
    "DevOps": "apply DevOps culture, flow and feedback practices",
    "Monitoring": "instrument services and design dashboards and alerts",
    "Prometheus": "collect and query metrics with Prometheus and PromQL",
    "Grafana": "visualise metrics and build alerting dashboards in Grafana",
    "Logging": "centralise and search structured application logs",
    "Observability": "combine metrics, logs and traces to debug distributed systems",
    "Site Reliability Engineering": "define SLIs, SLOs and error budgets and reduce toil",
    "Incident Management": "run on-call rotations, incident response and blameless post-mortems",
    "Microservices": "decompose systems into services with API contracts and resilience patterns",
    "Service Mesh": "manage traffic, mTLS and telemetry with a service mesh",
    "Chaos Engineering": "run controlled failure experiments to validate resilience",
    "Platform Engineering": "build internal developer platforms and golden paths",
    "Deployment Strategies": "execute blue-green, canary and rolling releases safely",
    # web
    "HTML": "structure accessible pages with semantic HTML",
    "CSS": "style layouts with CSS flexbox and grid",
    "Responsive Design": "build mobile-first responsive layouts",
    "React": "build component-based user interfaces with React hooks",
    "State Management": "manage client state with Redux-style stores and context",
    "Frontend Development": "assemble, bundle and ship production front-end applications",
    "Next.js": "build routed React applications with Next.js",
    "Server-Side Rendering": "render pages on the server for performance and SEO",
    "Angular": "build enterprise single-page applications with Angular",
    "Vue.js": "build reactive interfaces with Vue components and composition API",
    "Node.js": "build server-side applications with Node.js and npm",
    "Express": "build HTTP servers and middleware with Express",
    "Backend Development": "design service layers, persistence and background jobs for web back ends",
    "REST APIs": "design and implement resource-oriented REST APIs with proper status codes",
    "API Design": "version, document and evolve public APIs with OpenAPI",
    "FastAPI": "build typed asynchronous APIs with FastAPI and Pydantic",
    "Django": "build database-backed web applications with Django ORM and templates",
    "Flask": "build small web services and blueprints with Flask",
    "Spring Boot": "build production-ready Spring Boot services with dependency injection",
    "GraphQL": "design GraphQL schemas and resolvers",
    "Authentication": "implement sessions, tokens and multi-factor authentication",
    "OAuth": "integrate OAuth 2.0 and OpenID Connect flows",
    "Web Performance": "measure and improve Core Web Vitals and load times",
    "Web Accessibility": "audit and fix pages against WCAG guidelines",
    "UX Design": "run user research, wireframing and usability tests",
    "React Native": "build cross-platform mobile apps with React Native",
    "Mobile Development": "structure, test and publish mobile applications",
    "WebSockets": "build real-time features with WebSockets",
    "Jest": "write component and integration tests with Jest",
    # security
    "Cybersecurity": "explain core security principles, threats and controls",
    "Network Security": "configure firewalls, segmentation and intrusion detection",
    "Application Security": "find and remediate common web application vulnerabilities",
    "OWASP": "apply the OWASP Top 10 to code review and testing",
    "Secure Coding": "write input-validated, least-privilege code",
    "Cryptography": "apply symmetric, asymmetric and hashing primitives correctly",
    "Penetration Testing": "plan and execute authorised penetration tests and write reports",
    "Threat Modeling": "model threats with STRIDE and data-flow diagrams",
    "Security Operations": "triage alerts in a security operations centre",
    "SIEM": "write detections and dashboards in a SIEM",
    "Incident Response": "contain, eradicate and recover from security incidents",
    "Digital Forensics": "acquire and analyse disk and memory evidence",
    "DevSecOps": "embed SAST, DAST and dependency scanning in CI/CD",
    "Vulnerability Management": "scan, prioritise and track vulnerability remediation",
    "Compliance": "map controls to SOC 2 and ISO 27001 requirements",
    "Risk Management": "identify, score and treat information risks",
    "Container Security": "harden images, runtimes and clusters",
    # professional skills
    "Communication": "communicate clearly and concisely with technical and non-technical colleagues",
    "Teamwork": "collaborate effectively in cross-functional teams",
    "Collaboration": "coordinate asynchronous work across distributed teams",
    "Technical Writing": "write clear technical documentation and design documents",
    "Presentation Skills": "structure and deliver persuasive presentations",
    "Agile": "apply agile principles, backlogs and iterative delivery",
    "Scrum": "run Scrum ceremonies and roles",
    "Project Management": "plan scope, schedule, budget and risks for projects",
    "Leadership": "lead teams through ambiguity and give effective feedback",
    "Mentoring": "coach and develop junior engineers",
    "People Management": "manage performance, hiring and career growth for a team",
    "Stakeholder Management": "align expectations and manage stakeholders",
    "Product Management": "define product strategy, roadmaps and prioritisation",
    "Business Analysis": "analyse business processes and recommend solutions",
    "Critical Thinking": "evaluate evidence and arguments rigorously",
    "Negotiation": "negotiate scope, deadlines and trade-offs",
    "Time Management": "prioritise work and manage deadlines",
}

# --------------------------------------------------------------------------------------
# Course rows:  (area, title, level, hours, [taught skills], [prerequisite skills])
# Taught skills are listed most-central first.
# --------------------------------------------------------------------------------------
R = []
def c(area, title, level, hours, skills, prereqs=()):
    # a course never "teaches" a skill it lists as a prerequisite
    skills = [s for s in skills if s not in set(prereqs)]
    R.append((area, title, level, hours, list(skills), list(prereqs)))

# programming / SE ------------------------------------------------------------------
c("prog", "Python Programming Fundamentals", "beginner", 24, ["Python", "Debugging"])
c("prog", "Object-Oriented Programming in Python", "intermediate", 20, ["Object-Oriented Programming", "Python"], ["Python"])
c("prog", "Data Structures and Algorithms with Python", "intermediate", 40, ["Data Structures", "Algorithms"], ["Python"])
c("prog", "Advanced Algorithms and Complexity", "advanced", 36, ["Algorithms", "Dynamic Programming", "Problem Solving"], ["Data Structures"])
c("prog", "Technical Interview Problem Solving", "intermediate", 20, ["Problem Solving", "Algorithms"], ["Data Structures"])
c("prog", "Java Programming Essentials", "beginner", 30, ["Java", "Object-Oriented Programming"])
c("prog", "Spring Boot Microservices", "intermediate", 32, ["Spring Boot", "Java", "Microservices"], ["Java", "REST APIs"])
c("prog", "JavaScript from Scratch", "beginner", 22, ["JavaScript"])
c("prog", "TypeScript for JavaScript Developers", "intermediate", 16, ["TypeScript"], ["JavaScript"])
c("prog", "Go for Backend Services", "intermediate", 28, ["Go", "Concurrency"])
c("prog", "C++ Systems Programming", "intermediate", 40, ["C++", "Data Structures"])
c("prog", "Rust for Systems Programmers", "advanced", 36, ["Rust", "Concurrency"], ["C++"])
c("prog", "Bash Scripting and the Command Line", "beginner", 14, ["Bash Scripting", "Linux"])
c("prog", "Linux Administration Essentials", "beginner", 24, ["Linux"])
c("prog", "Linux System Administration in Depth", "advanced", 30, ["Linux System Administration", "Bash Scripting"], ["Linux"])
c("prog", "Git and Collaborative Development", "beginner", 10, ["Git"])
c("prog", "Unit Testing and Test-Driven Development", "intermediate", 18, ["Unit Testing", "Test-Driven Development"])
c("prog", "Software Testing and QA Automation", "intermediate", 24, ["Software Testing", "Test Automation"])
c("prog", "Software Testing Basics", "beginner", 10, ["Software Testing"])
c("prog", "Design Patterns and Clean Code", "advanced", 26, ["Design Patterns", "Code Review"], ["Object-Oriented Programming"])
c("prog", "Software Architecture and System Design", "advanced", 32, ["Software Architecture", "System Design"], ["Object-Oriented Programming"])
c("prog", "Functional Programming Concepts", "intermediate", 18, ["Functional Programming"])
c("prog", "Computer Networking Fundamentals", "beginner", 22, ["Computer Networking", "TCP/IP"])
c("prog", "Debugging and Performance Profiling", "intermediate", 12, ["Debugging", "Performance Profiling"])
c("prog", "Concurrency and Parallel Programming", "advanced", 24, ["Concurrency", "Performance Profiling"], ["Data Structures"])
# data -------------------------------------------------------------------------------
c("data", "SQL for Data Analysis", "beginner", 20, ["SQL", "Data Analysis"])
c("data", "SQL Crash Course", "beginner", 6, ["SQL"])
c("data", "Relational Database Design and PostgreSQL", "intermediate", 28, ["PostgreSQL", "Database Design", "SQL"], ["SQL"])
c("data", "MySQL Administration", "intermediate", 20, ["MySQL", "Database Administration"], ["SQL"])
c("data", "Advanced SQL: Window Functions and Query Tuning", "advanced", 20, ["Query Optimization", "SQL"], ["SQL"])
c("data", "NoSQL Databases with MongoDB", "intermediate", 20, ["NoSQL", "MongoDB"])
c("data", "Redis and Caching Strategies", "intermediate", 12, ["Redis", "Caching"])
c("data", "Data Modeling and Dimensional Design", "intermediate", 22, ["Data Modeling", "Data Warehousing"], ["SQL"])
c("data", "Cloud Data Warehousing with Snowflake", "intermediate", 24, ["Snowflake", "Data Warehousing"], ["SQL"])
c("data", "ETL and Data Pipeline Engineering", "intermediate", 32, ["ETL", "Data Pipelines"], ["Python", "SQL"])
c("data", "Apache Airflow Workflow Orchestration", "intermediate", 20, ["Apache Airflow", "Data Pipelines"], ["Python"])
c("data", "Analytics Engineering with dbt", "intermediate", 18, ["dbt", "Data Modeling"], ["SQL"])
c("data", "Apache Spark for Big Data Processing", "advanced", 36, ["Apache Spark", "Big Data"], ["Python", "SQL"])
c("data", "Streaming Data with Apache Kafka", "advanced", 28, ["Apache Kafka", "Stream Processing"], ["Python"])
c("data", "Hadoop Ecosystem Fundamentals", "intermediate", 22, ["Hadoop", "Big Data"], ["Linux"])
c("data", "Pandas and NumPy for Data Wrangling", "beginner", 22, ["Pandas", "NumPy", "Data Cleaning"], ["Python"])
c("data", "Python for Data Science Crash Course", "beginner", 16, ["Python", "Pandas"])
c("data", "Data Visualization with Python", "intermediate", 18, ["Data Visualization", "Matplotlib"], ["Pandas"])
c("data", "Tableau for Business Analytics", "beginner", 16, ["Tableau", "Data Visualization"])
c("data", "Power BI Dashboards", "beginner", 16, ["Power BI", "Business Intelligence"])
c("data", "Excel for Data Analysis", "beginner", 14, ["Excel", "Data Analysis"])
c("data", "Probability and Statistics Foundations", "beginner", 30, ["Statistics", "Probability"])
c("data", "Applied Statistical Inference", "intermediate", 28, ["Hypothesis Testing", "Statistics"], ["Probability"])
c("data", "A/B Testing and Experiment Design", "intermediate", 18, ["A/B Testing", "Hypothesis Testing"], ["Statistics"])
c("data", "Exploratory Data Analysis in Practice", "intermediate", 20, ["Exploratory Data Analysis", "Data Analysis"], ["Pandas"])
c("data", "Data Engineering Capstone", "advanced", 40, ["Data Pipelines", "Data Warehousing", "ETL"], ["ETL", "SQL"])
c("data", "Business Intelligence and Reporting", "intermediate", 20, ["Business Intelligence", "Data Storytelling"])
c("data", "Data Governance and Quality", "intermediate", 16, ["Data Governance", "Data Quality"])
c("data", "R for Statistical Computing", "intermediate", 24, ["R", "Statistics"])
c("data", "Time Series Analysis and Forecasting", "advanced", 26, ["Time Series Analysis"], ["Statistics", "Python"])
c("data", "Data Analyst Career Track", "beginner", 40, ["SQL", "Excel", "Tableau", "Data Analysis"])
# ML / AI ----------------------------------------------------------------------------
c("ml", "Linear Algebra for Machine Learning", "beginner", 24, ["Linear Algebra"])
c("ml", "Calculus and Optimization for ML", "intermediate", 22, ["Calculus", "Optimization"], ["Linear Algebra"])
c("ml", "Machine Learning Foundations", "intermediate", 36, ["Machine Learning", "Scikit-Learn", "Model Evaluation"], ["Python", "Statistics", "Linear Algebra"])
c("ml", "Practical Machine Learning with Python", "intermediate", 24, ["Machine Learning", "Scikit-Learn"], ["Python"])
c("ml", "Feature Engineering and Model Selection", "intermediate", 20, ["Feature Engineering", "Model Evaluation"], ["Machine Learning"])
c("ml", "Ensemble Methods and Gradient Boosting", "advanced", 20, ["Ensemble Methods", "Gradient Boosting"], ["Machine Learning"])
c("ml", "Unsupervised Learning and Clustering", "intermediate", 18, ["Unsupervised Learning"], ["Machine Learning"])
c("ml", "Anomaly Detection Methods", "advanced", 20, ["Anomaly Detection"], ["Machine Learning"])
c("ml", "Deep Learning with PyTorch", "advanced", 40, ["Deep Learning", "PyTorch", "Neural Networks"], ["Machine Learning", "Python"])
c("ml", "Deep Learning with TensorFlow and Keras", "advanced", 40, ["Deep Learning", "TensorFlow", "Neural Networks"], ["Machine Learning", "Python"])
c("ml", "Deep Learning Overview for Practitioners", "beginner", 12, ["Deep Learning", "Neural Networks"])
c("ml", "Computer Vision with Convolutional Networks", "advanced", 36, ["Computer Vision"], ["Deep Learning"])
c("ml", "NLP Fundamentals", "intermediate", 28, ["Natural Language Processing"], ["Python", "Machine Learning"])
c("ml", "Transformer Architectures and Attention", "advanced", 32, ["Transformers", "Natural Language Processing"], ["Deep Learning", "PyTorch"])
c("ml", "Prompt Engineering for Developers", "intermediate", 10, ["Prompt Engineering", "Generative AI"], ["Python"])
c("ml", "Large Language Models in Practice", "advanced", 30, ["Large Language Models", "Prompt Engineering"], ["Python"])
c("ml", "Fine-Tuning Large Language Models", "advanced", 28, ["Fine-Tuning", "Large Language Models"], ["Transformers"])
c("ml", "Retrieval-Augmented Generation Systems", "advanced", 24, ["Retrieval-Augmented Generation", "Vector Databases", "Large Language Models"], ["Prompt Engineering"])
c("ml", "Recommender Systems", "advanced", 28, ["Recommender Systems"], ["Machine Learning"])
c("ml", "Reinforcement Learning Essentials", "advanced", 32, ["Reinforcement Learning"], ["Deep Learning"])
c("ml", "MLOps: Deploying and Monitoring Models", "advanced", 32, ["MLOps", "Model Deployment", "Model Monitoring"], ["Machine Learning", "Docker"])
c("ml", "ML Experiment Tracking and Pipelines", "intermediate", 16, ["MLOps", "Experiment Tracking"], ["Machine Learning", "Python"])
c("ml", "Model Serving and Inference Optimization", "advanced", 24, ["Model Deployment", "Inference Optimization"], ["Machine Learning"])
c("ml", "Responsible AI and Model Fairness", "intermediate", 14, ["Responsible AI"], ["Machine Learning"])
c("ml", "Applied Data Science Capstone", "advanced", 40, ["Data Science", "Machine Learning", "Data Analysis"], ["Machine Learning", "Pandas"])
c("ml", "Introduction to AI and Data Science", "beginner", 10, ["Artificial Intelligence", "Data Science"])
c("ml", "Generative AI for Business Professionals", "beginner", 8, ["Generative AI"])
c("ml", "ML Engineer Career Track", "advanced", 60, ["Machine Learning", "Model Deployment", "MLOps", "Docker", "Python"], ["Python"])
# cloud ------------------------------------------------------------------------------
c("cloud", "Cloud Computing Concepts", "beginner", 10, ["Cloud Computing"])
c("cloud", "AWS Cloud Practitioner Essentials", "beginner", 14, ["Amazon Web Services", "Cloud Computing"])
c("cloud", "AWS Solutions Architecture", "intermediate", 36, ["Amazon Web Services", "Cloud Architecture"], ["Amazon Web Services"])
c("cloud", "AWS Serverless Applications", "intermediate", 22, ["AWS Lambda", "Serverless"], ["Amazon Web Services"])
c("cloud", "AWS Deep Dive Bootcamp", "advanced", 48, ["Amazon Web Services", "Cloud Architecture", "Serverless", "Cloud Security"], ["Cloud Computing"])
c("cloud", "Microsoft Azure Fundamentals", "beginner", 14, ["Microsoft Azure", "Cloud Computing"])
c("cloud", "Azure Solutions Architecture", "intermediate", 34, ["Microsoft Azure", "Cloud Architecture"], ["Microsoft Azure"])
c("cloud", "Google Cloud Platform Fundamentals", "beginner", 14, ["Google Cloud Platform", "Cloud Computing"])
c("cloud", "GCP Data and ML Engineering", "advanced", 32, ["Google Cloud Platform", "Model Deployment"], ["Google Cloud Platform", "Python"])
c("cloud", "Terraform Infrastructure as Code", "intermediate", 24, ["Terraform", "Infrastructure as Code"], ["Cloud Computing"])
c("cloud", "Ansible Configuration Management", "intermediate", 18, ["Ansible", "Configuration Management"], ["Linux"])
c("cloud", "Cloud Networking and VPC Design", "intermediate", 20, ["Cloud Networking"], ["Cloud Computing", "Computer Networking"])
c("cloud", "Cloud Cost Optimization (FinOps)", "intermediate", 12, ["Cloud Cost Optimization"], ["Cloud Computing"])
c("cloud", "Multi-Cloud Architecture and Resilience", "advanced", 28, ["Cloud Architecture", "High Availability"], ["Cloud Architecture"])
c("cloud", "Identity and Access Management in the Cloud", "intermediate", 14, ["Identity and Access Management", "Cloud Security"], ["Cloud Computing"])
c("cloud", "Cloud Engineer Career Track", "intermediate", 48, ["Amazon Web Services", "Terraform", "Linux", "Cloud Networking"])
# devops -----------------------------------------------------------------------------
c("devops", "Docker and Containers Fundamentals", "beginner", 16, ["Docker", "Containerization"], ["Linux"])
c("devops", "Docker Crash Course", "beginner", 6, ["Docker"])
c("devops", "Kubernetes Administration", "advanced", 36, ["Kubernetes", "Container Orchestration"], ["Docker"])
c("devops", "Kubernetes Quickstart", "beginner", 8, ["Kubernetes"])
c("devops", "Helm and Kubernetes Application Delivery", "advanced", 16, ["Helm"], ["Kubernetes"])
c("devops", "CI/CD Pipelines with GitHub Actions", "intermediate", 16, ["CI/CD", "GitHub Actions"], ["Git"])
c("devops", "Jenkins Pipeline Automation", "intermediate", 18, ["Jenkins", "CI/CD"], ["Git"])
c("devops", "GitOps with Declarative Delivery", "advanced", 14, ["GitOps"], ["Kubernetes"])
c("devops", "DevOps Culture and Practices", "beginner", 12, ["DevOps"])
c("devops", "Monitoring with Prometheus and Grafana", "intermediate", 18, ["Prometheus", "Grafana", "Monitoring"], ["Linux"])
c("devops", "Logging and Observability", "intermediate", 18, ["Observability", "Logging"], ["Monitoring"])
c("devops", "Site Reliability Engineering Principles", "advanced", 28, ["Site Reliability Engineering", "Incident Management"], ["Monitoring"])
c("devops", "Microservices Architecture Patterns", "advanced", 30, ["Microservices", "Software Architecture"], ["REST APIs"])
c("devops", "Service Mesh Essentials", "advanced", 16, ["Service Mesh"], ["Kubernetes"])
c("devops", "Chaos Engineering and Resilience Testing", "advanced", 12, ["Chaos Engineering"], ["Site Reliability Engineering"])
c("devops", "Platform Engineering Fundamentals", "advanced", 24, ["Platform Engineering", "Infrastructure as Code"], ["Kubernetes", "CI/CD"])
c("devops", "Release Engineering and Deployment Strategies", "intermediate", 14, ["Deployment Strategies"], ["CI/CD"])
c("devops", "DevOps Engineer Career Track", "intermediate", 50, ["Docker", "CI/CD", "Linux", "Terraform", "Kubernetes"])
# web --------------------------------------------------------------------------------
c("web", "HTML and CSS Foundations", "beginner", 16, ["HTML", "CSS"])
c("web", "Responsive Web Design", "intermediate", 14, ["Responsive Design", "CSS"], ["CSS"])
c("web", "React Fundamentals", "intermediate", 26, ["React", "Frontend Development"], ["JavaScript"])
c("web", "Advanced React and State Management", "advanced", 24, ["React", "State Management"], ["React"])
c("web", "Next.js Full-Stack Applications", "advanced", 24, ["Next.js", "Server-Side Rendering"], ["React"])
c("web", "Angular Application Development", "intermediate", 30, ["Angular", "TypeScript"], ["JavaScript"])
c("web", "Vue.js Essentials", "intermediate", 20, ["Vue.js"], ["JavaScript"])
c("web", "Node.js and Express Backend", "intermediate", 26, ["Node.js", "Express", "Backend Development"], ["JavaScript"])
c("web", "REST API Design and Development", "intermediate", 20, ["REST APIs", "API Design"])
c("web", "FastAPI for Python Web Services", "intermediate", 18, ["FastAPI", "REST APIs"], ["Python"])
c("web", "Django Web Development", "intermediate", 30, ["Django", "Backend Development"], ["Python"])
c("web", "Flask Microframework", "beginner", 12, ["Flask"], ["Python"])
c("web", "GraphQL APIs", "intermediate", 16, ["GraphQL"], ["REST APIs"])
c("web", "Web Authentication and OAuth", "intermediate", 14, ["Authentication", "OAuth"], ["REST APIs"])
c("web", "Web Performance Optimization", "advanced", 14, ["Web Performance"], ["JavaScript"])
c("web", "Web Accessibility (WCAG)", "intermediate", 10, ["Web Accessibility"], ["HTML"])
c("web", "UX Design Fundamentals", "beginner", 18, ["UX Design"])
c("web", "React Native Mobile Apps", "advanced", 28, ["React Native", "Mobile Development"], ["React"])
c("web", "WebSockets and Real-Time Applications", "advanced", 14, ["WebSockets"], ["Node.js"])
c("web", "Frontend Testing with Jest and Cypress", "intermediate", 14, ["Jest", "Test Automation"], ["JavaScript"])
c("web", "Full-Stack Web Development Bootcamp", "intermediate", 60, ["Frontend Development", "Backend Development", "JavaScript", "Node.js", "REST APIs"])
c("web", "Backend Engineer Career Track", "intermediate", 55, ["Backend Development", "SQL", "REST APIs", "Docker", "Python"])
# security ---------------------------------------------------------------------------
c("sec", "Cybersecurity Fundamentals", "beginner", 18, ["Cybersecurity"])
c("sec", "Network Security Essentials", "intermediate", 24, ["Network Security"], ["Computer Networking"])
c("sec", "Application Security and the OWASP Top 10", "intermediate", 20, ["Application Security", "OWASP"])
c("sec", "Secure Coding Practices", "intermediate", 16, ["Secure Coding"])
c("sec", "Applied Cryptography", "advanced", 26, ["Cryptography"])
c("sec", "Ethical Hacking and Penetration Testing", "advanced", 34, ["Penetration Testing"], ["Network Security", "Linux"])
c("sec", "Threat Modeling", "intermediate", 12, ["Threat Modeling"], ["Cybersecurity"])
c("sec", "Security Operations and SIEM", "intermediate", 24, ["Security Operations", "SIEM"], ["Cybersecurity"])
c("sec", "Incident Response and Digital Forensics", "advanced", 26, ["Incident Response", "Digital Forensics"], ["Security Operations"])
c("sec", "Cloud Security Architecture", "advanced", 24, ["Cloud Security", "Cloud Architecture"], ["Cloud Computing", "Cybersecurity"])
c("sec", "DevSecOps Pipelines", "advanced", 20, ["DevSecOps"], ["CI/CD", "Cybersecurity"])
c("sec", "Vulnerability Management", "intermediate", 14, ["Vulnerability Management"], ["Cybersecurity"])
c("sec", "Security Compliance: SOC 2 and ISO 27001", "intermediate", 14, ["Compliance", "Risk Management"])
c("sec", "Container and Kubernetes Security", "advanced", 16, ["Container Security"], ["Docker"])
c("sec", "Identity and Access Management Fundamentals", "beginner", 12, ["Identity and Access Management"])
c("sec", "Cybersecurity Analyst Career Track", "beginner", 50, ["Cybersecurity", "Network Security", "Security Operations", "Incident Response"])
# professional skills ----------------------------------------------------------------
c("soft", "Effective Communication for Technical Teams", "beginner", 8, ["Communication", "Teamwork"])
c("soft", "Technical Writing and Documentation", "beginner", 10, ["Technical Writing"])
c("soft", "Presentation and Public Speaking", "beginner", 8, ["Presentation Skills"])
c("soft", "Data Storytelling", "intermediate", 10, ["Data Storytelling", "Presentation Skills"])
c("soft", "Agile and Scrum Fundamentals", "beginner", 10, ["Agile", "Scrum"])
c("soft", "Project Management Foundations", "beginner", 24, ["Project Management"])
c("soft", "Technical Leadership and Mentoring", "advanced", 16, ["Leadership", "Mentoring"])
c("soft", "Engineering Management Essentials", "advanced", 20, ["Leadership", "People Management"])
c("soft", "Stakeholder Management", "intermediate", 10, ["Stakeholder Management"])
c("soft", "Product Management Essentials", "intermediate", 20, ["Product Management"])
c("soft", "Business Analysis and Requirements Engineering", "intermediate", 20, ["Business Analysis", "Requirements Engineering"])
c("soft", "Critical Thinking and Problem Solving", "beginner", 8, ["Critical Thinking", "Problem Solving"])
c("soft", "Negotiation and Conflict Resolution", "intermediate", 8, ["Negotiation"])
c("soft", "Time Management and Productivity", "beginner", 6, ["Time Management"])
c("soft", "Remote Team Collaboration", "beginner", 6, ["Collaboration", "Communication"])

AREA_NAME = {
    "prog": "Programming", "data": "Data", "ml": "AI & ML", "cloud": "Cloud",
    "devops": "DevOps", "web": "Web", "sec": "Security", "soft": "Professional Skills",
}


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def h(cid: str, salt: str) -> float:
    return int(hashlib.sha256((cid + salt).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


CASE_SENSITIVE = {"R", "Go"}


def real_df(postings_path: Path | None):
    """Optional: count real job postings mentioning each skill (word-boundary, lowercase)."""
    if not postings_path or not postings_path.exists():
        return {}
    skills = sorted({s for r in R for s in r[4]} | {p for r in R for p in r[5]})
    alias = {
        "Amazon Web Services": ["amazon web services", "aws"], "Microsoft Azure": ["azure"],
        "Google Cloud Platform": ["gcp", "google cloud"], "CI/CD": ["ci/cd", "continuous integration"],
        "Machine Learning": ["machine learning"], "SQL": ["sql"], "Large Language Models": ["llm", "large language model"],
        "Natural Language Processing": ["nlp", "natural language processing"], "Scikit-Learn": ["scikit-learn", "sklearn"],
        "Kubernetes": ["kubernetes", "k8s"], "Node.js": ["node.js", "nodejs"], "Vue.js": ["vue"],
        "REST APIs": ["rest api", "restful"], "Apache Spark": ["spark"], "Apache Kafka": ["kafka"],
        "Apache Airflow": ["airflow"], "Infrastructure as Code": ["infrastructure as code", "iac"],
    }
    pats = {}
    for s in skills:
        terms = alias.get(s, [s.lower()])
        if s in CASE_SENSITIVE:
            pats[s] = re.compile(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9+#])")
        else:
            pats[s] = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in terms) + r")(?![a-z0-9+#])")
    cnt = {s: 0 for s in skills}
    n = 0
    for line in postings_path.open(encoding="utf-8"):
        raw = json.loads(line).get("job_description_text", "")
        text = raw.lower()
        n += 1
        for s, p in pats.items():
            # one/two-letter skill names (R, Go) must match case-sensitively to avoid stop-word hits
            if p.search(raw if s in CASE_SENSITIVE else text):
                cnt[s] += 1
    for noisy in ("R", "Go"):  # single-token names are too ambiguous for a naive text match
        cnt[noisy] = None
    print(f"[real-data grounding] scanned {n} real postings (R/Go df not reported: ambiguous)")
    return cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "inputs/courses/course_catalog_large.jsonl"))
    ap.add_argument("--postings", default=str(ROOT.parent / "SkillLLM_v2/Extract_jobDesc/data/job_descriptions.jsonl"))
    args = ap.parse_args()

    df = real_df(Path(args.postings))
    taught_by: dict[str, list[str]] = {}
    counters: dict[str, int] = {}
    rows = []
    missing_ph = set()
    for area, title, level, hours, skills, prereqs in R:
        counters[area] = counters.get(area, 0) + 1
        cid = f"{area}-{counters[area]:03d}"
        for s in skills:
            taught_by.setdefault(s, []).append(cid)
        missing_ph.update(s for s in skills if s not in PH)
        outcomes = [f"Able to {PH[s]}" for s in skills[:4] if s in PH]
        art = "An" if level[0] in "ai" else "A"
        desc = (f"{title}. {art} {level} course ({hours} hours) in which learners {PH[skills[0]] if skills[0] in PH else skills[0]}"
                + (f", and also {', '.join(PH[s] for s in skills[1:3] if s in PH)}" if len(skills) > 1 else "")
                + ".")
        rating = round(3.9 + 1.0 * h(cid, "r"), 1)
        cost = 0.0 if h(cid, "c") < 0.35 else round(10 + 90 * h(cid, "p"), 0)
        rows.append({
            "course_id": cid,
            "title": title,
            "description": desc,
            "learning_outcomes": outcomes,
            "teaches_skills": skills,
            "prerequisites": prereqs,
            "level": level,
            "duration_hours": hours,
            "provider": f"Synthetic {AREA_NAME[area]} Academy",
            "rating": rating,
            "cost_usd": cost,
            "url": None,
            "source": "synthetic",
            "metadata": {
                "source": "synthetic",
                "area": area,
                "generator": "scripts/build_course_catalog_large.py",
                "note": "Fictional provider; hand-authored structure + templated outcomes. Not a real course.",
                "real_posting_df": {s: df.get(s, None) for s in skills} if df else None,
            },
        })
    # validation: every prerequisite is taught by some course; no self-prereq
    orphan = sorted({p for r in R for p in r[5] if p not in taught_by})
    selfp = [r[1] for r in R if set(r[5]) & set(r[4])]
    print(f"courses={len(rows)} unique_taught_skills={len(taught_by)} orphan_prereqs={orphan} self_prereq={selfp}")
    if missing_ph:
        print("WARNING skills without phrase:", sorted(missing_ph))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_area = {}
    for r in rows:
        by_area[r["metadata"]["area"]] = by_area.get(r["metadata"]["area"], 0) + 1
    lv = {}
    for r in rows:
        lv[r["level"]] = lv.get(r["level"], 0) + 1
    print("by area:", by_area, "by level:", lv)
    if df:
        covered = [s for s in taught_by if (df.get(s) or 0) > 0]
        print(f"skills with >=1 real posting mention: {len(covered)}/{len(taught_by)}")
        top = sorted(taught_by, key=lambda s: -(df.get(s) or 0))[:15]
        print("top vocab skills by real posting df:", [(s, df[s]) for s in top])
    print("wrote", out)


if __name__ == "__main__":
    main()

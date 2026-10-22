"""Hand-authored (user, job) scenarios for the course-recommendation benchmark.

This file is DATA.  It is the single place where scenario ground truth is authored; nothing here is
computed from any recommender's output.

Each scenario S(...) records
  * user.known   - skills the user demonstrably has (proficient)
  * user.weak    - skills the user has but only at beginner level (still needs work)
  * job text     - what the *text-based* baselines see (written in natural language, sometimes with
                   aliases such as AWS / k8s / sklearn that differ from the catalogue vocabulary)
  * req / pref   - the explicit skill lists given to systems that consume structured jobs
                   (what a skill extractor would pull out of the posting)
  * implicit     - skills the posting text implies but does NOT list explicitly.  They are part of the
                   gold need but are *not* passed to any system as structured input; this deliberately
                   makes the gold need set differ from the systems' input lists.

Gold need weights (used only by build_relevance_labels.py): req = 3, implicit = 2, pref = 1.

Provenance: ALL 48 scenarios are SYNTHETIC.  Scenarios 1-41 are archetypes (typical career-transition /
upskilling situations).  Scenarios 42-48 originally used excerpts of real job postings as the job text; those
excerpts (and every company / source-row reference) were removed for privacy and replaced by the synthetic
job descriptions below (``source = "synthetic_rewrite"``), written from the authored skill lists only.  The
req / pref / implicit skill annotations and the user profiles of 42-48 are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _l(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@dataclass
class S:
    sid: str
    background: str
    level: str  # junior | mid | senior  (user experience level)
    known: str
    job_title: str
    text: str
    req: str
    pref: str = ""
    implicit: str = ""
    weak: str = ""
    source: str = "synthetic"
    provenance: dict | None = None  # note only; never a company / source file / row index

    def known_list(self) -> list[str]:
        return _l(self.known)

    def weak_list(self) -> list[str]:
        return _l(self.weak)


# ---- synthetic job descriptions for scenarios 42-48 (written from the authored skill lists only) ----
_PRIVACY_NOTE = {"note": "replaces a real-posting excerpt that was removed for privacy; text is synthetic"}
_TEXT_42 = (
    "We are looking for an intern to join a small team that analyses commodity market data. You will pull trading and pricing records from a relational database using SQL and clean them with Python and the pandas library. Day to day you will compute summary statistics, spot unusual price movements and work out what drives them. Findings are shared as charts and short written notes that traders and risk colleagues can read quickly. You will document your queries so teammates can rerun them. We expect a curious student who likes numbers, asks good questions and is comfortable learning on the job. An interest in energy or agricultural markets is welcome."
)
_TEXT_43 = (
    "This internship places you in an applied AI group that prototypes machine learning features for internal business tools. You will train and test deep learning models, mostly with TensorFlow, and run experiments on the Azure cloud platform. Part of the role is trying out language-model ideas, such as text classification and summarisation, and building a small React and TypeScript demo to show the results to colleagues. You will measure how well each model performs and report where it fails. We are looking for a student with solid Python and Git habits and a willingness to read papers and discuss trade-offs. You will work closely with a mentor and present your findings at the end of the term."
)
_TEXT_44 = (
    "The intern will help a manufacturing engineering team predict when production equipment is likely to need servicing. You will explore sensor and log data, build machine learning models for failure prediction and tune them with optimisation methods. A strong mathematical foundation is important, since you will reason about loss functions, constraints and uncertainty. Experience with PyTorch is a plus. You will compare candidate models, write up what worked and what did not, and prepare clear technical notes for the engineers who maintain the machines. This is a research-flavoured placement for a master's student who enjoys turning theory into working tools and is comfortable with Python and numerical computing."
)
_TEXT_45 = (
    "Join the quality team as a software test engineering intern and help us check embedded and desktop software before it ships. You will write automated test scripts, extend unit tests written by developers and keep everything in a shared Git repository. You will also help connect the tests to our build pipeline so they run on every change, which means picking up some DevOps and continuous integration practice. When something fails, you will investigate, reproduce the problem and file a clear report. We are looking for a student with some C++ or Python experience who is careful, patient and likes finding the edge cases that others miss. Mentoring is provided throughout."
)
_TEXT_46 = (
    "We are hiring a junior product manager to guide a software product used by technical customers. You will talk to users, collect their needs and turn them into clear requirements for the engineering team. You will look at usage data to decide what to build next and to check whether released features helped. Working with stakeholders in sales, support and engineering, you will balance competing requests and keep everyone informed about priorities. Being able to read code and query a PostgreSQL database is useful, and familiarity with TypeScript front ends helps you follow technical discussions. We value clear writing, good judgement and a hands-on attitude."
)
_TEXT_47 = (
    "The associate data engineer supports a team that moves and prepares data for analysts across the business. You will build and monitor ETL jobs and data pipelines, check incoming data for gaps or errors and fix quality problems at their source. Pipelines should write useful logs so that issues can be traced quickly. You will also write short technical documents describing how each dataset is produced, and help refresh the reports in our business intelligence tool. A background in Python and SQL is expected, and spreadsheet skills are a plus. We are looking for someone organised and careful who enjoys making data dependable for other people."
)
_TEXT_48 = (
    "This internship sits in a go-to-market operations group that is adding AI to its everyday work. You will build data pipelines that gather customer and sales information into a central warehouse, writing SQL to shape the tables that business intelligence dashboards rely on. You will also prototype generative AI helpers, writing and testing prompts that draft summaries or answer questions about accounts. Reporting the results with simple visuals is part of the job. We are looking for a student comfortable with Python who learns quickly, communicates well and is happy to work with sales and marketing teammates to understand what they need."
)


SCENARIOS: list[S] = [
    S("sc-01", "Junior Python developer", "junior", "Python,Git,Linux",
      "ML Platform Engineer",
      "Seeking an ML Platform Engineer to build containerized machine learning pipelines on AWS with Docker and Kubernetes, and to automate training releases.",
      "Docker,Machine Learning,Amazon Web Services", "Kubernetes,CI/CD", "MLOps"),
    S("sc-02", "Data analyst moving to ML", "mid", "Python,SQL,Statistics,Linear Algebra,Pandas",
      "Machine Learning Engineer",
      "Machine Learning Engineer to develop predictive models with sklearn, build neural network architectures with PyTorch and deploy models in production. Strong grasp of evaluation methodology expected.",
      "Machine Learning,Scikit-Learn,Deep Learning,PyTorch", "MLOps", "Model Evaluation"),
    S("sc-03", "Front-end developer going full stack", "mid", "JavaScript,React,TypeScript,HTML,CSS",
      "Full Stack Cloud Engineer",
      "Full stack engineer building Python REST APIs with FastAPI, relational Postgres schemas, containerized with Docker and deployed on AWS.",
      "Python,FastAPI,PostgreSQL,Docker", "Amazon Web Services", "REST APIs"),
    S("sc-04", "Software engineer into cloud architecture", "mid", "Python,Git,Linux,Bash Scripting",
      "Cloud & DevOps Architect",
      "Architect multi-tier cloud infrastructure on AWS with automated CI/CD pipelines, Docker containerization and Kubernetes (k8s) cluster orchestration. Infrastructure is managed declaratively.",
      "Amazon Web Services,Docker,Kubernetes,CI/CD", "Cloud Architecture", "Terraform"),
    S("sc-05", "CS graduate into NLP", "junior", "Python,Data Structures,Algorithms,Linear Algebra,Statistics",
      "NLP & LLM Research Engineer",
      "Seeking an NLP engineer to fine-tune large language models, implement transformer architectures in PyTorch and evaluate token classification pipelines.",
      "Deep Learning,PyTorch,Transformers,Large Language Models", "Natural Language Processing", "Machine Learning"),
    S("sc-06", "Marketing analyst becoming a data analyst", "junior", "Excel,Statistics,Data Visualization,Communication",
      "Data Analyst",
      "We need a Data Analyst who can pull data with SQL, script analyses in Python, and publish dashboards in Tableau for the growth team. Experience with experiments is a plus.",
      "SQL,Python,Tableau", "Data Storytelling,A/B Testing", "Pandas,Data Analysis"),
    S("sc-07", "Accountant moving to BI", "junior", "Excel,Communication,Risk Management,Presentation Skills",
      "Business Intelligence Analyst",
      "Build and maintain Power BI reports for finance leadership. Write SQL against the data mart, define KPIs and keep the semantic model consistent.",
      "Power BI,SQL,Business Intelligence", "Data Storytelling", "Data Modeling"),
    S("sc-08", "Linux sysadmin to DevOps", "mid", "Linux,Bash Scripting,Computer Networking,Linux System Administration",
      "DevOps Engineer",
      "DevOps engineer to containerize services, own the delivery pipeline, provision environments as code and operate our Kubernetes clusters. Comfortable with config management.",
      "Docker,CI/CD,Terraform,Kubernetes", "Prometheus,Ansible", "Git"),
    S("sc-09", "Java backend developer to data engineering", "mid", "Java,Spring Boot,SQL,Git,REST APIs,Object-Oriented Programming",
      "Data Engineer",
      "Data engineer needed to build Spark jobs in Python and robust ETL into our warehouse; you will schedule workflows with Airflow and consume Kafka topics.",
      "Apache Spark,Python,ETL", "Apache Airflow,Apache Kafka", "Data Pipelines"),
    S("sc-10", "Manual QA tester to SDET", "mid", "Software Testing,Git,SQL",
      "Software Development Engineer in Test",
      "SDET to write automated regression suites in Python, add unit tests alongside developers and wire everything into the CI pipeline running in containers.",
      "Test Automation,Python,Unit Testing", "CI/CD,Docker", "Test-Driven Development"),
    S("sc-11", "Data scientist to MLOps", "mid", "Python,Machine Learning,Scikit-Learn,Pandas,Statistics,Linear Algebra,Model Evaluation",
      "MLOps Engineer",
      "MLOps engineer to productionize models: package them in containers, serve them on Kubernetes, and automate retraining with CI. Command-line fluency on Linux servers required.",
      "MLOps,Docker,Kubernetes,Model Deployment", "CI/CD,Experiment Tracking", "Linux"),
    S("sc-12", "Security analyst to cloud security", "mid", "Cybersecurity,Network Security,Computer Networking,Security Operations",
      "Cloud Security Engineer",
      "Cloud Security Engineer responsible for hardening AWS accounts, designing least-privilege IAM policies and shifting security left in delivery pipelines.",
      "Cloud Security,Amazon Web Services,Identity and Access Management", "DevSecOps,Terraform", "Cloud Computing"),
    S("sc-13", "Project coordinator to data/AI product manager", "mid", "Agile,Scrum,Communication,Stakeholder Management",
      "Technical Product Manager (Data & AI)",
      "Own the roadmap for our analytics platform. You will query data yourself, read experiment results with engineers, and prioritise features with stakeholders.",
      "SQL,Data Analysis,Product Management", "A/B Testing", "Statistics"),
    S("sc-14", "Bootcamp grad to Python backend", "junior", "HTML,CSS,JavaScript,Node.js,Git",
      "Backend Engineer (Python)",
      "Backend engineer building Django services on PostgreSQL, exposing REST endpoints; Docker and Redis caching are used in our stack.",
      "Python,Django,PostgreSQL,REST APIs", "Docker,Redis", "Backend Development"),
    S("sc-15", "Data engineer to streaming platform", "senior", "Python,SQL,ETL,Apache Airflow,Data Pipelines",
      "Streaming Data Platform Engineer",
      "Platform engineer for our real-time data platform: Kafka, Spark streaming, running on Kubernetes. You operate petabyte-scale event flows.",
      "Apache Kafka,Apache Spark,Stream Processing", "Kubernetes,Docker", "Big Data"),
    S("sc-16", "Physics graduate to data scientist", "junior", "Python,NumPy,Linear Algebra,Calculus,Statistics,Probability",
      "Data Scientist",
      "Data Scientist to build and validate predictive models with scikit-learn, query our data lake with SQL and present results with clear visuals.",
      "Machine Learning,Pandas,SQL,Data Visualization", "Deep Learning", "Scikit-Learn,Feature Engineering"),
    S("sc-17", "C++ embedded developer to Rust systems", "mid", "C++,Linux,Data Structures,Algorithms,Debugging",
      "Rust Systems Engineer",
      "Systems engineer writing high-performance, memory-safe network services in Rust. Deep understanding of protocols and parallel code is essential.",
      "Rust,Concurrency,Computer Networking", "Docker", "TCP/IP"),
    S("sc-18", "React Native developer to Next.js full stack", "mid", "JavaScript,React,React Native,TypeScript,Mobile Development",
      "Full-Stack Engineer (Next.js)",
      "Full-stack engineer on a Next.js application with a GraphQL API on Node, PostgreSQL storage and OAuth sign-in. SSR performance matters.",
      "Next.js,Node.js,GraphQL", "PostgreSQL,Authentication", "Server-Side Rendering,Backend Development"),
    S("sc-19", "DevOps engineer to SRE", "senior", "Docker,Kubernetes,CI/CD,Linux,Git,Jenkins,Containerization",
      "Site Reliability Engineer",
      "SRE to define SLOs, build alerting on Prometheus metrics, run incident response and reduce toil through automation and infrastructure as code.",
      "Site Reliability Engineering,Prometheus,Observability,Incident Management", "Terraform", "Monitoring,Grafana"),
    S("sc-20", "AWS cloud engineer to multi-cloud architect", "senior", "Amazon Web Services,Cloud Computing,Linux,Terraform,Infrastructure as Code",
      "Multi-Cloud Solutions Architect",
      "Design resilient multi-cloud platforms across Azure and Google Cloud as well as AWS, with strong cost governance and failover design.",
      "Microsoft Azure,Google Cloud Platform,Cloud Architecture", "Cloud Cost Optimization", "High Availability"),
    S("sc-21", "CS student to new-grad SWE", "junior", "Python,Java,Data Structures,Algorithms,Git",
      "Software Engineer (New Grad)",
      "New-grad software engineer on a backend team: write well-tested services, apply design patterns, use SQL databases and ship containers.",
      "Unit Testing,Design Patterns,SQL", "REST APIs,Docker", "System Design"),
    S("sc-22", "BI analyst to analytics engineer", "mid", "SQL,Excel,Tableau,Data Analysis",
      "Analytics Engineer",
      "Analytics engineer to own our transformation layer in dbt on Snowflake, with version-controlled models and python utilities.",
      "dbt,Data Modeling,Python", "Snowflake,Apache Airflow", "Git"),
    S("sc-23", "ML engineer to LLM application developer", "mid", "PyTorch,Deep Learning,Python,Machine Learning,Neural Networks",
      "LLM Application Engineer",
      "Build LLM-powered assistants: retrieval over company documents with embeddings, careful prompt design, and a FastAPI serving layer.",
      "Large Language Models,Retrieval-Augmented Generation,Prompt Engineering", "Vector Databases,FastAPI", "Fine-Tuning"),
    S("sc-24", "ML researcher to computer vision engineer", "mid", "Deep Learning,PyTorch,Python,Linear Algebra,Calculus,Neural Networks,Machine Learning",
      "Computer Vision Engineer",
      "Computer vision engineer shipping detection models to edge devices: containerised inference services and latency tuning.",
      "Computer Vision,Model Deployment,Docker", "Inference Optimization", "MLOps"),
    S("sc-25", "Web designer to front-end developer", "junior", "HTML,CSS,UX Design,Responsive Design",
      "Front-End Developer",
      "Front-end developer building accessible, fast React interfaces in TypeScript with a version-controlled workflow.",
      "JavaScript,React,TypeScript", "Web Accessibility,Web Performance", "Frontend Development,Git"),
    S("sc-26", "IT support technician to cloud support engineer", "junior", "Computer Networking,Linux,TCP/IP",
      "Cloud Support Engineer",
      "Cloud support engineer: troubleshoot customer workloads on AWS, VPC connectivity issues and Linux instances; automate fixes with infrastructure as code.",
      "Amazon Web Services,Linux System Administration,Cloud Networking", "Terraform", "Cloud Computing"),
    S("sc-27", "Junior pentester to application security", "mid", "Cybersecurity,Network Security,Linux,Penetration Testing",
      "Application Security Engineer",
      "AppSec engineer to review code and threat-model new features, find OWASP-class flaws and script tooling in Python.",
      "Application Security,OWASP,Secure Coding", "Threat Modeling,Python", "Vulnerability Management"),
    S("sc-28", "Business graduate to business analyst", "junior", "Excel,Communication,Presentation Skills",
      "Business Analyst",
      "Business analyst to elicit requirements from stakeholders, query data with SQL and visualise findings in Power BI; works in an agile team.",
      "Business Analysis,SQL,Requirements Engineering", "Power BI,Agile", "Data Analysis"),
    S("sc-29", "Senior Java engineer to engineering manager", "senior", "Java,Spring Boot,Microservices,SQL,Docker,Git,Software Architecture",
      "Engineering Manager",
      "Engineering manager for a payments platform team: grow engineers, run performance cycles and manage delivery with product and business stakeholders.",
      "Leadership,People Management,Mentoring", "Project Management,Stakeholder Management", "Communication"),
    S("sc-30", "Python web developer to data engineer", "mid", "Python,Flask,SQL,Git,REST APIs",
      "Data Engineer",
      "Data engineer to design our cloud warehouse, build large-scale Spark transformations and orchestrate daily loads.",
      "Apache Spark,ETL,Data Warehousing", "Snowflake,Apache Airflow", "Data Pipelines,Big Data"),
    S("sc-31", "R analyst to ML engineer", "mid", "R,Statistics,Probability,Excel,Data Visualization",
      "Machine Learning Engineer",
      "ML engineer to build tabular and deep learning models in Python and take them from notebooks to repeatable training code.",
      "Python,Machine Learning,Scikit-Learn", "Deep Learning", "Pandas"),
    S("sc-32", "Kubernetes admin to platform engineer", "senior", "Kubernetes,Docker,Linux,CI/CD,Git,Containerization,Container Orchestration",
      "Platform Engineer",
      "Platform engineer building an internal developer platform: Terraform modules, Helm-based delivery and declarative GitOps promotion across environments.",
      "Platform Engineering,Infrastructure as Code,Terraform", "GitOps,Helm,Service Mesh", "Cloud Computing"),
    S("sc-33", "Node.js developer to Go backend", "mid", "JavaScript,Node.js,Express,REST APIs,MongoDB,NoSQL,Git",
      "Backend Engineer (Go)",
      "Go backend engineer for high-throughput services: concurrency, PostgreSQL, caching hot paths and splitting monoliths into microservices.",
      "Go,Concurrency,PostgreSQL", "Microservices,Docker", "Redis,Caching"),
    S("sc-34", "Aspiring analyst with shaky SQL", "junior", "Excel,Data Analysis,Statistics",
      "Data Analyst",
      "Data analyst to work in SQL every day, wrangle data in Python and build dashboards for operations teams.",
      "SQL,Python,Data Visualization", "Tableau", "Pandas", weak="SQL"),
    S("sc-35", "Growth marketer to product analyst", "mid", "Excel,Data Visualization,Communication,Data Analysis",
      "Product Analyst",
      "Product analyst to run and read A/B tests, size effects with proper statistics, and query event data in SQL.",
      "SQL,A/B Testing,Hypothesis Testing", "Python,Tableau", "Statistics"),
    S("sc-36", "DevOps engineer to DevSecOps", "senior", "Docker,Kubernetes,CI/CD,Linux,Git,Jenkins",
      "DevSecOps Engineer",
      "DevSecOps engineer to add scanning and policy enforcement to delivery pipelines, secure container images and cluster workloads, and manage vulnerabilities.",
      "DevSecOps,Container Security,Vulnerability Management", "Terraform,Cloud Security", "Cybersecurity"),
    S("sc-37", "Django developer to production ML engineer", "mid", "Python,Django,PostgreSQL,REST APIs,Git,Docker",
      "Machine Learning Engineer (Production)",
      "ML engineer to ship models behind APIs and monitor them after launch; you will tune models built with sklearn and run them on Kubernetes.",
      "Machine Learning,Model Deployment,MLOps", "Scikit-Learn,Kubernetes", "Statistics,Linear Algebra"),
    S("sc-38", "Statistician to data scientist", "mid", "Statistics,Probability,R,Hypothesis Testing,Time Series Analysis",
      "Data Scientist",
      "Data scientist forecasting demand and churn in Python, pulling features from SQL warehouses and engineering features for tree-based and neural models.",
      "Python,Machine Learning,SQL", "Deep Learning,Pandas", "Feature Engineering"),
    S("sc-39", "Network engineer to cloud network engineer", "mid", "Computer Networking,TCP/IP,Linux,Network Security,Bash Scripting",
      "Cloud Network Engineer",
      "Cloud network engineer: design VPCs and hybrid connectivity on AWS, automate with Terraform, some Azure exposure.",
      "Cloud Networking,Amazon Web Services,Terraform", "Microsoft Azure", "Cloud Computing"),
    S("sc-40", "Self-taught web beginner (weak JS)", "junior", "Git,HTML,CSS,JavaScript,Python",
      "Full-Stack Developer",
      "Junior full-stack developer: React front end, Node.js services, SQL storage; TypeScript and Docker appreciated.",
      "JavaScript,React,Node.js,SQL", "TypeScript,Docker", "Frontend Development,Backend Development", weak="JavaScript"),
    S("sc-41", "API developer to serverless backend", "mid", "Python,Docker,Git,REST APIs,Amazon Web Services",
      "Serverless Backend Engineer",
      "Serverless backend engineer building event-driven Lambda services provisioned through infrastructure as code with good cost awareness.",
      "AWS Lambda,Serverless,Infrastructure as Code", "Terraform,Cloud Architecture", "Cloud Computing,Cloud Cost Optimization"),
    # ---- scenarios 42-48: formerly real-posting excerpts; job text is now synthetic (see _TEXT_* above) ----
    S("sc-42", "Business student to commodities data analyst intern", "junior", "Excel,Statistics,Communication,Presentation Skills",
      "Commodities Data Analyst Intern",
      _TEXT_42,
      "Python,SQL,Pandas", "Data Visualization", "Data Analysis", source="synthetic_rewrite", provenance=_PRIVACY_NOTE),
    S("sc-43", "CS student to AI intern (cloud/LLM)", "junior", "Python,Git,Data Structures,Algorithms,JavaScript",
      "AI Intern (cloud and LLM)",
      _TEXT_43,
      "Microsoft Azure,Machine Learning,Deep Learning,TensorFlow", "Natural Language Processing,React,TypeScript,Large Language Models", "Model Evaluation", source="synthetic_rewrite", provenance=_PRIVACY_NOTE),
    S("sc-44", "Applied-maths MSc to AI engineering intern", "junior", "Python,NumPy,Linear Algebra,Calculus,Probability,Statistics",
      "AI Engineering Intern: predictive maintenance",
      _TEXT_44,
      "Optimization,Machine Learning", "PyTorch,Technical Writing", "Model Evaluation", source="synthetic_rewrite", provenance=_PRIVACY_NOTE),
    S("sc-45", "Software engineering student to test engineering intern", "junior", "C++,Python,Object-Oriented Programming",
      "Software Test Engineering Intern",
      _TEXT_45,
      "Software Testing,Test Automation,Git", "DevOps,CI/CD", "Unit Testing", source="synthetic_rewrite", provenance=_PRIVACY_NOTE),
    S("sc-46", "CS graduate to junior product manager", "junior", "Python,JavaScript,React,Git,SQL",
      "Product Manager",
      _TEXT_46,
      "Product Management,Requirements Engineering,Data Analysis", "TypeScript,PostgreSQL", "Stakeholder Management", source="synthetic_rewrite", provenance=_PRIVACY_NOTE),
    S("sc-47", "Analyst to data engineering associate", "junior", "Python,SQL,Excel",
      "Associate Data Engineer",
      _TEXT_47,
      "ETL,Data Pipelines,Data Quality", "Business Intelligence,Technical Writing", "Logging", source="synthetic_rewrite", provenance=_PRIVACY_NOTE),
    S("sc-48", "Operations analyst to GTM data/AI intern", "junior", "Python,Excel,Communication",
      "Data Engineer Intern, GTM AI",
      _TEXT_48,
      "SQL,Business Intelligence,Prompt Engineering", "Data Warehousing,Generative AI", "Data Visualization", source="synthetic_rewrite", provenance=_PRIVACY_NOTE),
]

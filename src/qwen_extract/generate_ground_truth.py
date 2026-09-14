"""
Generates ground-truth structured Resume objects with randomized, realistic values.

Ground truth is built programmatically (not by an LLM) so it is always exactly correct —
the LLM is only used downstream (see textify.py) to turn this structured data into messy
prose, never to invent the labels themselves.
"""

from __future__ import annotations

import random

from faker import Faker

from qwen_extract.schema import Education, Experience, Resume

fake = Faker()

SKILLS_POOL = [
    "Python", "SQL", "Java", "JavaScript", "TypeScript", "AWS", "Azure", "Docker",
    "Kubernetes", "React", "Node.js", "Project Management", "Data Analysis",
    "Machine Learning", "C++", "Go", "Terraform", "CI/CD", "Agile", "Scrum",
    "Excel", "Tableau", "PowerBI", "REST APIs", "GraphQL", "Linux", "Git",
]

DEGREES = [
    "B.S. Computer Science", "B.A. Business Administration", "M.S. Data Science",
    "B.S. Electrical Engineering", "MBA", "B.A. Economics", "M.S. Computer Science",
    "B.S. Mathematics", "Associate Degree in Information Technology",
]

JOB_TITLES = [
    "Software Engineer", "Data Analyst", "Product Manager", "Business Analyst",
    "DevOps Engineer", "Data Scientist", "QA Engineer", "Marketing Coordinator",
    "Financial Analyst", "Operations Manager", "Sales Associate", "Systems Administrator",
]


def _random_date_pair() -> tuple[str, str]:
    start = fake.date_between(start_date="-15y", end_date="-2y")
    is_current = random.random() < 0.25
    if is_current:
        end_label = "Present"
        end_date = fake.date_between(start_date=start, end_date="today")
    else:
        end_date = fake.date_between(start_date=start, end_date="today")
        end_label = end_date.strftime("%b %Y")
    return start.strftime("%b %Y"), end_label


def generate_resume() -> Resume:
    """Produce one random, internally-consistent Resume ground-truth object."""
    name = fake.name()
    email = fake.email()
    phone = fake.phone_number()

    num_skills = random.randint(3, 7)
    skills = random.sample(SKILLS_POOL, num_skills)

    num_edu = random.randint(1, 2)
    education = [
        Education(
            degree=random.choice(DEGREES),
            institution=fake.company() + " University" if random.random() < 0.5 else fake.city() + " College",
            year=str(random.randint(2005, 2023)),
        )
        for _ in range(num_edu)
    ]

    num_exp = random.randint(1, 4)
    experience = []
    total_months = 0
    for _ in range(num_exp):
        start_label, end_label = _random_date_pair()
        experience.append(
            Experience(
                title=random.choice(JOB_TITLES),
                company=fake.company(),
                start_date=start_label,
                end_date=end_label,
                description=fake.sentence(nb_words=10),
            )
        )
        total_months += random.randint(6, 48)

    total_years = round(total_months / 12, 1)

    return Resume(
        name=name,
        email=email,
        phone=phone,
        skills=skills,
        education=education,
        experience=experience,
        total_years_experience=total_years,
    )


def generate_batch(n: int, seed: int | None = None) -> list[Resume]:
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    return [generate_resume() for _ in range(n)]

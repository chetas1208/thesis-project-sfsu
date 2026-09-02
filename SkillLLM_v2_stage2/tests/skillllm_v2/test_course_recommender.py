from skillllm_v2.match import CourseRecommender


def test_course_recommender() -> None:
    recs = CourseRecommender().recommend([{"skill": "AWS", "gap_type": "missing_critical"}], [{"course_id": "c1", "course_name": "AWS", "teaches_skills": ["AWS"]}])
    assert recs[0]["course_id"] == "c1"

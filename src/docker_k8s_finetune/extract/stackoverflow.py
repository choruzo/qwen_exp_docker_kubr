from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from ..errors import ExtractionError
from ..schema import InterimRecord
from .base import BaseExtractor, ExtractionResult


QUERY = r"""
WITH candidate_answers AS (
  SELECT
    q.id AS question_id,
    q.title,
    q.body AS question_body,
    q.tags,
    q.score AS question_score,
    q.accepted_answer_id,
    a.id AS answer_id,
    a.body AS answer_body,
    a.score AS answer_score,
    a.owner_user_id,
    u.display_name,
    ROW_NUMBER() OVER (
      PARTITION BY q.id
      ORDER BY IF(a.id = q.accepted_answer_id, 1, 0) DESC, a.score DESC, a.id ASC
    ) AS answer_rank
  FROM `bigquery-public-data.stackoverflow.posts_questions` q
  JOIN `bigquery-public-data.stackoverflow.posts_answers` a
    ON a.parent_id = q.id
  LEFT JOIN `bigquery-public-data.stackoverflow.users` u
    ON u.id = a.owner_user_id
  WHERE q.score >= @min_question_score
    AND a.score >= @min_answer_score
    AND EXISTS (
      SELECT 1 FROM UNNEST(@tags) tag
      WHERE q.tags LIKE CONCAT('%<', tag, '>%')
    )
)
SELECT * EXCEPT(answer_rank)
FROM candidate_answers
WHERE answer_rank = 1
ORDER BY question_score DESC, answer_score DESC, question_id ASC
LIMIT @max_records
"""


class StackOverflowBigQueryExtractor(BaseExtractor):
    def _rows(self) -> Iterator[Any]:
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise ExtractionError("google-cloud-bigquery is required for Stack Overflow extraction") from exc
        project_env = str(self.source_config.get("project_env", "GOOGLE_CLOUD_PROJECT"))
        project = os.environ.get(project_env)
        if not project:
            raise ExtractionError(f"Set {project_env} to a billed Google Cloud project for BigQuery extraction")
        client = bigquery.Client(project=project)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("tags", "STRING", self.source_config.get("tags", [])),
                bigquery.ScalarQueryParameter("min_question_score", "INT64", self.source_config.get("min_question_score", 1)),
                bigquery.ScalarQueryParameter("min_answer_score", "INT64", self.source_config.get("min_answer_score", 1)),
                bigquery.ScalarQueryParameter("max_records", "INT64", self.source_config.get("max_records", 20000)),
            ]
        )
        yield from client.query(QUERY, job_config=job_config).result()

    def _records(self) -> Iterator[InterimRecord]:
        for row in self._rows():
            question_id = int(row["question_id"])
            answer_id = int(row["answer_id"])
            question = f"{row['title']}\n\n{row['question_body']}".strip()
            answer = str(row["answer_body"]).strip()
            url = f"https://stackoverflow.com/questions/{question_id}"
            yield InterimRecord(
                source=self.source_name,
                category="troubleshooting",
                raw_content=f"USER:\n{question}\n\nASSISTANT:\n{answer}",
                license="CC-BY-SA-4.0",
                license_url=str(self.source_config["license_url"]),
                url=url,
                source_record_id=f"q{question_id}-a{answer_id}",
                title=str(row["title"]),
                author=str(row["display_name"] or row["owner_user_id"] or "deleted-user"),
                attribution=(
                    f"Stack Overflow question {question_id}, answer {answer_id}, "
                    f"author {row['display_name'] or row['owner_user_id'] or 'deleted-user'}"
                ),
                language="en",
                metadata={
                    "normalization": "direct_pair",
                    "pair": {"user": question, "assistant": answer},
                    "question_score": int(row["question_score"]),
                    "answer_score": int(row["answer_score"]),
                    "owner_user_id": row["owner_user_id"],
                    "display_name": row["display_name"],
                    "tags": str(row["tags"]),
                },
            )

    def extract(self) -> ExtractionResult:
        if self.should_skip():
            return self.skipped_result()
        return self.write_records(self._records())

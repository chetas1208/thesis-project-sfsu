---
license: cc-by-4.0
language: en
---

This is the SkillSpan dataset created by:

```
@inproceedings{zhang-etal-2022-skillspan,
    title = "{S}kill{S}pan: Hard and Soft Skill Extraction from {E}nglish Job Postings",
    author = "Zhang, Mike  and
      Jensen, Kristian  and
      Sonniks, Sif  and
      Plank, Barbara",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.366",
    doi = "10.18653/v1/2022.naacl-main.366",
    pages = "4962--4984"
}
```

There are document delimiters indicated by `idx`.

Number of samples (sentences):
- train: 4800
- dev: 3174
- test: 3569

Sources:
- Stackoverflow (tech)
- STAR (house)

Type of tags:
- Generic BIO tags with keys `tags_skill` and `tags_knowledge`

Sample:
```
{
  "idx": 53, 
  "tokens": ["Drive", "our", "IT", "compliance", "agenda", "and", "develop", "our", "processes"], 
  "tags_skill": ["B", "I", "I", "I", "I", "O", "B", "I", "I"], 
  "tags_knowledge": ["O", "O", "O", "O", "O", "O", "O", "O", "O"], 
  "source": "house"
}
```
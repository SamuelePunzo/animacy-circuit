from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ANIMATE_ENTITIES = [
    "Michael Jackson",
    "Taylor Swift",
    "Beyonce",
    "Rihanna",
    "Madonna",
    "Adele",
    "Elvis Presley",
    "Bob Dylan",
    "Freddie Mercury",
    "David Bowie",
    "Prince",
    "Lady Gaga",
    "Billie Eilish",
    "Kendrick Lamar",
    "Paul McCartney",
    "John Lennon",
    "Aretha Franklin",
    "Stevie Wonder",
    "Whitney Houston",
    "Mariah Carey",
    "Barack Obama",
    "Michelle Obama",
    "Angela Merkel",
    "Nelson Mandela",
    "Martin Luther King",
    "Mahatma Gandhi",
    "Winston Churchill",
    "Abraham Lincoln",
    "John Kennedy",
    "Franklin Roosevelt",
    "Theodore Roosevelt",
    "George Washington",
    "Napoleon Bonaparte",
    "Julius Caesar",
    "Cleopatra",
    "Elizabeth Windsor",
    "Charles Darwin",
    "Albert Einstein",
    "Isaac Newton",
    "Marie Curie",
    "Ada Lovelace",
    "Alan Turing",
    "Nikola Tesla",
    "Stephen Hawking",
    "Jane Goodall",
    "Rosalind Franklin",
    "Galileo Galilei",
    "Leonardo da Vinci",
    "Pablo Picasso",
    "Vincent van Gogh",
    "Frida Kahlo",
    "Claude Monet",
    "Andy Warhol",
    "Georgia O'Keeffe",
    "William Shakespeare",
    "Jane Austen",
    "Charles Dickens",
    "Virginia Woolf",
    "Toni Morrison",
    "George Orwell",
    "J.K. Rowling",
    "Agatha Christie",
    "Homer",
    "Dante Alighieri",
    "Leo Tolstoy",
    "Fyodor Dostoevsky",
    "Gabriel Garcia Marquez",
    "Haruki Murakami",
    "Chinua Achebe",
    "Oprah Winfrey",
    "Ellen DeGeneres",
    "David Attenborough",
    "Greta Thunberg",
    "Malala Yousafzai",
    "Serena Williams",
    "Venus Williams",
    "Naomi Osaka",
    "Simone Biles",
    "Usain Bolt",
    "Michael Jordan",
    "LeBron James",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Roger Federer",
    "Rafael Nadal",
    "Tiger Woods",
    "Tom Brady",
    "Muhammad Ali",
    "Meryl Streep",
    "Denzel Washington",
    "Morgan Freeman",
    "Tom Hanks",
    "Leonardo DiCaprio",
    "Robert De Niro",
    "Al Pacino",
    "Natalie Portman",
    "Emma Watson",
    "Scarlett Johansson",
    "Viola Davis",
    "Cate Blanchett",
    "Keanu Reeves",
    "Jackie Chan",
    "Bruce Lee",
    "Steven Spielberg",
    "Martin Scorsese",
    "Quentin Tarantino",
    "Christopher Nolan",
    "Hayao Miyazaki",
    "Stanley Kubrick",
    "Alfred Hitchcock",
]


INANIMATE_ENTITIES = [
    "Paris",
    "London",
    "Berlin",
    "Rome",
    "Madrid",
    "Amsterdam",
    "Brussels",
    "Vienna",
    "Prague",
    "Warsaw",
    "Lisbon",
    "Athens",
    "Dublin",
    "Copenhagen",
    "Stockholm",
    "Oslo",
    "Helsinki",
    "Reykjavik",
    "Zurich",
    "Geneva",
    "New York",
    "Los Angeles",
    "Chicago",
    "Boston",
    "Seattle",
    "Toronto",
    "Montreal",
    "Vancouver",
    "Mexico City",
    "Buenos Aires",
    "Sao Paulo",
    "Rio de Janeiro",
    "Tokyo",
    "Kyoto",
    "Osaka",
    "Seoul",
    "Beijing",
    "Shanghai",
    "Hong Kong",
    "Singapore",
    "Bangkok",
    "Mumbai",
    "Delhi",
    "Dubai",
    "Istanbul",
    "Cairo",
    "Nairobi",
    "Cape Town",
    "Sydney",
    "Melbourne",
    "Google",
    "Microsoft",
    "Apple",
    "Amazon",
    "Meta",
    "Netflix",
    "Disney",
    "Sony",
    "Samsung",
    "Toyota",
    "Tesla",
    "Ford",
    "Volkswagen",
    "Boeing",
    "Airbus",
    "NASA",
    "UNESCO",
    "NATO",
    "FIFA",
    "Harvard",
    "Stanford",
    "Oxford",
    "Cambridge",
    "Wikipedia",
    "YouTube",
    "Instagram",
    "Facebook",
    "Twitter",
    "TikTok",
    "iPhone",
    "Android",
    "Windows",
    "macOS",
    "Linux",
    "Chrome",
    "Firefox",
    "Photoshop",
    "Excel",
    "PowerPoint",
    "Minecraft",
    "Fortnite",
    "Wikipedia",
    "Wikipedia",
    "Star Wars",
    "Harry Potter",
    "Lord of the Rings",
    "Game of Thrones",
    "The Matrix",
    "Titanic",
    "Avatar",
    "Hamlet",
    "Macbeth",
    "Mona Lisa",
    "Guernica",
    "Wikipedia",
    "Mount Everest",
    "Sahara",
    "Amazon River",
    "Nile",
    "Pacific Ocean",
    "Atlantic Ocean",
]


ENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .'-]*[A-Za-z0-9]$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    return repo_root() / "dataset" / "semantic_meaningful" / "named_entity_targets.json"


def default_report_path() -> Path:
    return repo_root() / "dataset" / "semantic_meaningful" / "named_entity_grammar_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create named-entity target sets.")
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--report", type=Path, default=default_report_path())
    return parser.parse_args()


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def validate_entities(animate: list[str], inanimate: list[str]) -> dict:
    overlap = sorted(set(animate) & set(inanimate))
    invalid = {
        "animate": [item for item in animate if ENTITY_RE.fullmatch(item) is None],
        "inanimate": [item for item in inanimate if ENTITY_RE.fullmatch(item) is None],
    }
    return {
        "counts": {
            "animate": len(animate),
            "inanimate": len(inanimate),
            "animate_unique": len(set(animate)),
            "inanimate_unique": len(set(inanimate)),
        },
        "overlap": overlap,
        "invalid_surface_forms": invalid,
    }


def grammar_examples(entities: list[str], prefixes: list[str], terminal: str) -> list[str]:
    examples: list[str] = []
    for prefix in prefixes:
        if not prefix.endswith(" by the"):
            continue
        stem = prefix.removesuffix(" by the")
        for entity in entities[:3]:
            if terminal == "by_the":
                examples.append(f"{stem} by the {entity}")
            else:
                examples.append(f"{stem} by {entity}")
            if len(examples) >= 9:
                return examples
    return examples


def build_report(payload: dict) -> str:
    animate = payload["targets"]["animate"]
    inanimate = payload["targets"]["inanimate"]
    counts = payload["validation"]["counts"]
    subtypes = Counter(row["entity_type"] for row in payload["records"]["inanimate"])
    return "\n".join(
        [
            "# Named Entity Grammar Report",
            "",
            "## Target Counts",
            "",
            f"- Animate named entities: {counts['animate']}",
            f"- Inanimate/non-person named entities: {counts['inanimate']}",
            f"- Inanimate subtype counts: {dict(sorted(subtypes.items()))}",
            "",
            "## Surface Grammar Judgment",
            "",
            "- `by the <named entity>` is generally ungrammatical for ordinary proper names.",
            "- `by <named entity>` is the correct surface form for people, places, organizations, products, and works.",
            "- Organizations such as `Google` are grammatically good passive agents in `by Google`.",
            "- Locations such as `Paris` are grammatically well-formed in `by Paris`, but often require metonymic interpretation as a government, institution, or place-associated actor.",
            "",
            "## Examples: Current Template",
            "",
            *[f"- {item}" for item in payload["grammar_examples"]["by_the_animate"]],
            *[f"- {item}" for item in payload["grammar_examples"]["by_the_inanimate"]],
            "",
            "## Examples: Named-Entity Template",
            "",
            *[f"- {item}" for item in payload["grammar_examples"]["bare_by_animate"]],
            *[f"- {item}" for item in payload["grammar_examples"]["bare_by_inanimate"]],
            "",
            "## Recommendation",
            "",
            "Use the bare `by` template for named-entity completions and do not feed these lists into the current single-token common-noun target metric without a multi-token sequence-probability metric.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    animate = unique_preserving_order(ANIMATE_ENTITIES)
    inanimate = unique_preserving_order(INANIMATE_ENTITIES)
    sample_prefixes = [
        "The manuscript was annotated by the",
        "The tunnel was secured by the",
        "The database was archived by the",
    ]
    payload = {
        "meta": {
            "script": "create_named_entity_target_sets.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_source": "named_entities",
            "template_policy": "remove terminal article: use 'by <entity>', not 'by the <entity>'",
            "metric_warning": (
                "Many named entities are multi-token strings. The current single-token "
                "average-logit target metric is not valid for this artifact without adaptation."
            ),
        },
        "targets": {
            "animate": animate,
            "inanimate": inanimate,
        },
        "records": {
            "animate": [{"surface": item, "entity_type": "PERSON"} for item in animate],
            "inanimate": [
                {
                    "surface": item,
                    "entity_type": (
                        "GPE_OR_LOCATION"
                        if index < 50
                        else "ORG"
                        if index < 80
                        else "PRODUCT_OR_WORK"
                    ),
                }
                for index, item in enumerate(inanimate)
            ],
        },
        "validation": validate_entities(animate, inanimate),
        "grammar_examples": {
            "by_the_animate": grammar_examples(animate, sample_prefixes, "by_the"),
            "by_the_inanimate": grammar_examples(inanimate, sample_prefixes, "by_the"),
            "bare_by_animate": grammar_examples(animate, sample_prefixes, "bare_by"),
            "bare_by_inanimate": grammar_examples(inanimate, sample_prefixes, "bare_by"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report.write_text(build_report(payload), encoding="utf-8")
    print(f"Saved named entity targets to {args.output}")
    print(f"Saved grammar report to {args.report}")


if __name__ == "__main__":
    main()

import json
import csv
from pathlib import Path
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# --- Configuration ---

# The script's location is the base for finding the input file.
# It navigates up two directories, then into 'data/evaluate/'.
SCRIPT_DIR = Path(__file__).parent
INPUT_JSON_FILE = SCRIPT_DIR / "../../data/evaluate/raw_inference_resultsV3.json"
OUTPUT_CSV_FILE = INPUT_JSON_FILE.parent / "manuel_metricsV3.csv"

# --- Main Application ---

def clear_screen():
    """Clears the terminal screen for a cleaner interface."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_validated_score(prompt_text: str) -> int:
    """Prompts the user for a score between 1 and 5 and validates it."""
    try:
        score = Prompt.ask(prompt_text, choices=[str(i) for i in range(1, 6)])
        return int(score)
    except KeyboardInterrupt:
        # Allows graceful exit if user presses Ctrl+C during prompt
        console.print("\n[yellow]Evaluation interrupted. Exiting.[/yellow]")
        sys.exit()

def get_validated_citation(prompt_text: str) -> str:
    """Prompts the user for a Yes/No answer and validates it."""
    try:
        # Using 'e' and 'h' for Evet/Hayır for faster input
        answer = Prompt.ask(prompt_text, choices=["e", "h"], default="e")
        return "Yes" if answer.lower() == 'e' else "No"
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted. Exiting.[/yellow]")
        sys.exit()

def process_evaluation(item: dict, console: Console) -> dict:
    """Displays a single item and gets the user's evaluation for it."""
    clear_screen()

    # Display the information in a clean, readable panel
    query_text = Text(f"Query: {item.get('query', 'N/A')}", style="bold cyan")
    expected_answer_text = Text(f"Expected Answer: {item.get('expected_answer', 'N/A')}", style="green")
    generated_answer_text = Text(f"Generated Answer: {item.get('generated_answer', 'N/A')}", style="yellow")

    display_panel = Panel(
        Text("\n\n").join([query_text, expected_answer_text, generated_answer_text]),
        title="[bold]Question for Evaluation[/bold]",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(display_panel)
    console.print("Please provide your evaluation below. Press Ctrl+C to exit at any time.")

    # Get validated inputs from the user
    relevance = get_validated_score("Relevance (İlgililik)")
    accuracy = get_validated_score("Accuracy (Doğruluk)")
    source_citation = get_validated_citation("Source Citation (Kaynak Belirtme) [e/h]")
    fluency = get_validated_score("Fluency (Akıcılık)")

    return {
        "Relevance": relevance,
        "Accuracy": accuracy,
        "Source_Citation": source_citation,
        "Fluency": fluency,
    }

def main():
    """Main function to run the evaluation script."""
    global console
    console = Console()

    # 1. Check if the input JSON file exists
    if not INPUT_JSON_FILE.is_file():
        console.print(f"[bold red]Error: Input file not found at the expected path:[/bold red]")
        console.print(f"[cyan]{INPUT_JSON_FILE.resolve()}[/cyan]")
        console.print("Please ensure the file exists and the script is in the correct location.")
        return

    # 2. Load the questions from the JSON file
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        if not isinstance(questions, list):
            console.print("[bold red]Error: JSON file content is not a list of questions.[/bold red]")
            return
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Could not decode JSON from {INPUT_JSON_FILE.name}. Please check its format.[/bold red]")
        return

    # 3. Prepare and run the evaluation loop
    csv_file_exists = OUTPUT_CSV_FILE.is_file()
    
    try:
        with open(OUTPUT_CSV_FILE, mode='a', newline='', encoding='utf-8') as csv_file:
            # Define CSV columns, including original data for context.
            fieldnames = [
                "query", "expected_answer", "generated_answer",
                "Relevance", "Accuracy", "Source_Citation", "Fluency"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not csv_file_exists:
                writer.writeheader()

            total_items = len(questions)
            console.print(f"[bold green]Successfully loaded {total_items} questions from {INPUT_JSON_FILE.name}.[/bold green]")
            console.print(f"Evaluations will be saved to: [cyan]{OUTPUT_CSV_FILE.resolve()}[/cyan]")
            input("Press Enter to begin...")

            for index, item in enumerate(questions):
                console.print(f"\n--- Evaluating Item {index + 1} of {total_items} ---")
                
                evaluation_data = process_evaluation(item, console)
                
                row_data = {
                    "query": item.get("query"),
                    "expected_answer": item.get("expected_answer"),
                    "generated_answer": item.get("generated_answer"),
                    **evaluation_data
                }
                
                writer.writerow(row_data)
                csv_file.flush() # Ensure data is written immediately after each entry

    except KeyboardInterrupt:
        console.print(f"\n\n[bold yellow]Evaluation stopped by user. Progress has been saved to {OUTPUT_CSV_FILE.name}[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")
        console.print("Your progress so far has been saved.")

    console.print(f"\n[bold green]Evaluation complete! All data saved to {OUTPUT_CSV_FILE.name}[/bold green]")


if __name__ == "__main__":
    main()
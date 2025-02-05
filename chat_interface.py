# chat_interface.py
from rich.console import Console
from main import chat_agent  # Reuse chat logic

console = Console()

def main():
    console.rule("[bold magenta]Company Documents Chat[/]")
    console.print("[bold green]Type 'exit' to quit[/]\n")
    
    while True:
        query = console.input("[bold cyan]Your Question> [/]").strip()
        if query.lower() in ("exit", "quit"):
            break
        
        answer = chat_agent(query)
        console.print(f"\n[bold yellow]Answer:[/] {answer}\n")

if __name__ == "__main__":
    main()
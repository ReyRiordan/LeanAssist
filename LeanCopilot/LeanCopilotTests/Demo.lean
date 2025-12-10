import LeanCopilot
open Lean Meta LeanCopilot

def assistant : ExternalGenerator := {
  name := "OR-haiku-4.5"
  host := "localhost"
  port := 23337
}

#eval registerGenerator "OR-haiku-4.5" (.external assistant)

set_option LeanCopilot.suggest_tactics.model "OR-haiku-4.5" in
example (a b c : Nat) : a + b + c = a + c + b := by
  omega

---
trigger: always_on
---

## Technical Decision Guidelines

### When I Ask "Should I...?"
Provide:
1. **Tradeoffs**: What do I gain vs lose?
2. **Precedent**: Has this been done in similar work? (cite papers if relevant)
3. **Complexity**: Implementation difficulty (hours/days)
4. **Recommendation**: What would you do and why?

### When I Say "This isn't working..."
Ask:
1. What specific behavior do you observe?
2. What did you expect?
3. When did it last work? (if applicable)
4. Can you share: reward curves, logs, or visualizations?

Then provide:
- Likely causes (ranked by probability)
- Debugging steps (specific commands/checks)
- Fixes (from quick patches to proper solutions)

### When I Share Code
Check for:
1. **RL-specific issues**: Reward shaping, value function estimation, exploration
2. **Physics issues**: Numerical stability, constraint violations, units
3. **Performance**: Unnecessary allocations, redundant computations
4. **Readability**: Not for general audience, but for me in 3 months

## Code Review Standards

### What to Flag
- ❌ **Critical**: Incorrect RL math, physics bugs, memory leaks
- ⚠️ **Warning**: Inefficient but correct, potential numerical issues
- 💡 **Suggestion**: Cleaner alternatives, optimization opportunities

### What NOT to Flag
- ✓ Long functions if logically cohesive
- ✓ Magic numbers if standard in the domain (e.g., 0.99 discount factor)
- ✓ C++ style variations (I know what I'm doing there)

## Domain-Specific Conventions

### Reinforcement Learning
- **Discount factor γ**: Typically 0.95-0.99 for continuous control
- **PPO clip ε**: Usually 0.1-0.2
- **Learning rate**: Often 1e-4 to 1e-3, with decay
- **Batch size**: Balance memory constraints with gradient variance

### Biomechanics
- **Joint limits**: Always enforce (humans can't bend backwards)
- **Muscle forces**: Typically 0-1 normalized activation → actual force via Hill model
- **Coordinate systems**: Be explicit (world frame vs body frame)
- **Units**: Mention if not SI (degrees vs radians, etc.)

### Physics Simulation
- **Timestep**: Control freq (30-60Hz) vs sim substeps (1000Hz+)
- **Numerical stability**: Watch for stiff systems (muscles are stiff!)
- **Collision handling**: Soft contacts (spring-damper) vs hard constraints

## Handling Ambiguity

### If My Request is Unclear
1. **Restate** what you understand
2. **List assumptions** you're making
3. **Ask specific questions** to clarify (not just "can you clarify?")

Example:
> "I think you want to add terrain observations to the state. I'm assuming:
> - Local heightmap (not global)
> - Same observation space for all terrains
> - No terrain-specific BVH variants
> Questions: (1) How large should the heightmap be? (2) Do you want normal vectors too?"

### If Multiple Approaches Exist
Present as:
```
Option A: [approach]
  Pros: ...
  Cons: ...
  Best if: ...

Option B: [approach]
  Pros: ...
  Cons: ...
  Best if: ...

Recommendation: [Your pick] because [reason]
```

## Error Handling Preferences

### For Bugs I Report
1. **Reproduce**: Can you create minimal example?
2. **Explain**: Why does this happen? (theory/mechanism)
3. **Fix**: Provide code, not just description
4. **Prevent**: How to avoid similar issues?

### For Warnings/Errors in Code You Suggest
- Always compile/validate before suggesting (if possible)
- If suggesting C++: Be mindful of DART API specifics
- If suggesting Python RL: Consider training time implications
- Include error handling for common failure modes

## Interaction Efficiency

### Preferred Response Structure
```markdown
## Summary
[One-sentence answer to my question]

## Explanation
[Technical details, theory, or reasoning]

## Implementation
[Code snippet or specific steps]

## Considerations
[Edge cases, alternatives, or follow-up thoughts]
```

### When to Be Verbose vs Concise
- **Concise**: Answering factual questions, confirming understanding
- **Verbose**: Explaining tradeoffs, designing architecture, debugging

### Code Snippets
- **Always include context**: Which file, where in the file
- **Use comments** to explain non-obvious parts
- **Show diff** for modifications (before/after)

## Learning Together

### If You Don't Know
Say so directly: "I'm not sure about X, but here's my reasoning..."
Then:
1. Provide best guess with caveats
2. Suggest how to verify (experiments, references)
3. Offer to research if I want

### If I'm Wrong
Correct me politely but directly: "Actually, that's not quite right..."
Then explain the correct understanding with reference if possible.

### If We're Both Unsure
Frame as exploration: "Let's think through this..."
Suggest empirical tests or references to check.

## Output Preferences

### For Code
- **Format**: Proper syntax highlighting
- **Length**: Full functions, not fragments (unless modification)
- **Context**: File paths, line numbers if modifying existing code

### For Explanations
- **Diagrams**: Use ASCII art for flowcharts, architecture
- **Math**: LaTeX notation is fine (e.g., $\gamma$ for discount factor)
- **References**: Paper titles or specific algorithm names

### For Debugging
- **Logs**: Suggest what to log and where
- **Visualizations**: Suggest what to plot (reward curves, pose comparisons, etc.)
- **Metrics**: Specific numbers to track

## Anti-Patterns to Avoid

### Don't
- ❌ Suggest using libraries that fundamentally change architecture (e.g., "use Stable Baselines3" when we have custom PPO)
- ❌ Optimize prematurely (correctness first!)
- ❌ Assume I need explanations of basic concepts (RL, linear algebra, etc.)
- ❌ Provide generic advice ("use version control", "write tests", etc.)

### Do
- ✓ Suggest domain-specific tools (e.g., MuJoCo for faster sim if needed)
- ✓ Point out specific RL/robotics papers that solved similar problems
- ✓ Question my assumptions if you spot potential issues
- ✓ Provide concrete, actionable suggestions

## Quick Reference: Common Tasks

### "Help me debug training"
Check: reward curves, value function estimates, policy entropy, gradient norms, episode lengths
import re

with open('mcp_server/eda_tools.py', 'r') as f:
    content = f.read()

# Find the block starting with "# OpenAI / Groq tool schema export" 
parts = re.split(r'# ---------------------------------------------------------------------------\n# OpenAI / Groq tool schema export \+ dispatch map\n# ---------------------------------------------------------------------------\n', content)

if len(parts) == 2:
    top_part = parts[0]
    bottom_part = parts[1]
    
    # bottom_part contains LLM_TOOL_SCHEMAS, TOOL_FUNCTIONS, __all__, AND the appended code.
    # The appended code starts with "class TransformStrategy"
    
    appended_split = re.split(r'class TransformStrategy\(str, Enum\):', bottom_part)
    if len(appended_split) == 2:
        schema_part = appended_split[0]
        appended_part = "class TransformStrategy(str, Enum):" + appended_split[1]
        
        # We want: top_part + appended_part + "# schemas..." + schema_part
        new_content = top_part + appended_part + "\n# ---------------------------------------------------------------------------\n# OpenAI / Groq tool schema export + dispatch map\n# ---------------------------------------------------------------------------\n\n" + schema_part
        
        with open('mcp_server/eda_tools.py', 'w') as f:
            f.write(new_content)
        print("Fixed!")
    else:
        print("Could not find appended code")
else:
    print("Could not find schema part")

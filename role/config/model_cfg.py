"""
Model param configure
"""
param_cfg = {
    "gemma": {
        "template": "<start_of_turn>user \
                {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }}<end_of_turn> \
                <start_of_turn>model \
                {{ .Response }}<end_of_turn>",
        
        "penalize_newline": False,
        "repeat_penalty": 1,
        "stop": [
                "<start_of_turn>",
                "<end_of_turn>"
        ],
    },
    "wizardlm2": {
        "system": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        "stop": [
                "USER:",
                "ASSISTANT:"
            ],
        "template": "{{ if .System }}{{ .System }} {{ end }}{{ if .Prompt }}USER: {{ .Prompt }} {{ end }}ASSISTANT: {{ .Response }}",
    },
    "llama2": {
        "template": "[INST] <<SYS>>{{ .System }}<</SYS>>\
        \
        {{ .Prompt }} [/INST]",
        
        "stop": [
            "[INST]",
            "[/INST]",
            "<<SYS>>",
            "<</SYS>>"
        ]    
    },
    "llama3": {
        "template": """{{ if .System }}<|start_header_id|>system<|end_header_id|>

        {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

        {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

        {{ .Response }}<|eot_id|>""",
        
        "stop": [
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|reserved_special_token"
        ]    
    },
    "llava": {
        "template": """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]""",
        
        "stop": [
            "[INST]",
            "[/INST]"
        ]    
    },
}
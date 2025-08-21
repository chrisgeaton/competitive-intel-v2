"""
Analyze codebase for cleanup opportunities.
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict

def analyze_python_file(filepath):
    """Analyze a Python file for various metrics."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Count metrics
        imports = []
        functions = []
        classes = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(f"{node.module}.{node.names[0].name if node.names else '*'}")
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        # Check for potential issues
        issues = []
        
        # Unused imports detection (basic)
        for imp in imports:
            module_name = imp.split('.')[0]
            if module_name not in content:
                issues.append(f"Potentially unused import: {imp}")
        
        # Empty lines
        empty_lines = sum(1 for line in lines if not line.strip())
        
        # Unicode characters
        unicode_chars = re.findall(r'[^\x00-\x7F]', content)
        
        return {
            'filepath': filepath,
            'total_lines': len(lines),
            'code_lines': len(lines) - empty_lines,
            'empty_lines': empty_lines,
            'imports': imports,
            'functions': functions,
            'classes': classes,
            'issues': issues,
            'unicode_chars': len(set(unicode_chars)),
            'file_size': os.path.getsize(filepath)
        }
    
    except Exception as e:
        return {
            'filepath': filepath,
            'error': str(e)
        }

def analyze_codebase():
    """Analyze the entire codebase."""
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip venv and other directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'postgres-data']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print("CODEBASE ANALYSIS - BEFORE CLEANUP")
    print("=" * 50)
    
    total_files = len(python_files)
    total_lines = 0
    total_size = 0
    all_imports = []
    all_issues = []
    unicode_files = 0
    
    print(f"Python files found: {total_files}")
    print("\nFile-by-file analysis:")
    
    for filepath in sorted(python_files):
        analysis = analyze_python_file(filepath)
        
        if 'error' in analysis:
            print(f"ERROR analyzing {filepath}: {analysis['error']}")
            continue
        
        total_lines += analysis['total_lines']
        total_size += analysis['file_size']
        all_imports.extend(analysis['imports'])
        all_issues.extend(analysis['issues'])
        
        if analysis['unicode_chars'] > 0:
            unicode_files += 1
        
        print(f"  {filepath}")
        print(f"    Lines: {analysis['total_lines']} ({analysis['code_lines']} code, {analysis['empty_lines']} empty)")
        print(f"    Imports: {len(analysis['imports'])}")
        print(f"    Classes: {len(analysis['classes'])}, Functions: {len(analysis['functions'])}")
        print(f"    Size: {analysis['file_size']} bytes")
        if analysis['unicode_chars'] > 0:
            print(f"    Unicode chars: {analysis['unicode_chars']}")
        if analysis['issues']:
            print(f"    Issues: {len(analysis['issues'])}")
    
    # Summary
    import_counts = defaultdict(int)
    for imp in all_imports:
        import_counts[imp] += 1
    
    print(f"\nSUMMARY:")
    print(f"Total files: {total_files}")
    print(f"Total lines: {total_lines}")
    print(f"Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"Total imports: {len(all_imports)}")
    print(f"Unique imports: {len(set(all_imports))}")
    print(f"Files with Unicode: {unicode_files}")
    print(f"Potential issues: {len(all_issues)}")
    
    # Most common imports
    print(f"\nMost common imports:")
    for imp, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {imp}: {count} times")
    
    return {
        'files': total_files,
        'lines': total_lines,
        'size': total_size,
        'imports': len(all_imports),
        'unique_imports': len(set(all_imports)),
        'unicode_files': unicode_files,
        'issues': len(all_issues)
    }

if __name__ == "__main__":
    analyze_codebase()
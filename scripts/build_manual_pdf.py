#!/usr/bin/env python3
"""
Script de generación del Manual de Usuario en PDF para DS Guardian.
Combina la documentación de docs/ y genera un PDF estilizado mediante LibreOffice / Markdown.
"""

import os
import subprocess
import sys
import markdown

def build_pdf_manual():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_dir, 'docs')
    output_pdf = os.path.join(docs_dir, 'DS_Guardian_User_Manual.pdf')

    files_to_combine = [
        os.path.join(docs_dir, 'index.md'),
        os.path.join(docs_dir, 'getting-started.md'),
        os.path.join(docs_dir, 'installation.md'),
        os.path.join(docs_dir, 'tutorials/audit.md'),
        os.path.join(docs_dir, 'tutorials/eda.md'),
        os.path.join(docs_dir, 'tutorials/cleaning.md'),
        os.path.join(docs_dir, 'tutorials/models.md'),
        os.path.join(docs_dir, 'tutorials/visualization.md'),
        os.path.join(docs_dir, 'api.md'),
    ]

    combined_md = "# 🛡️ DS Guardian - Manual Completo de Usuario y Referencia Técnica\n\n"
    combined_md += "***AI Data Science Governance & Quality Assurance System***\n\n---\n\n"

    for fpath in files_to_combine:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                combined_md += f.read() + '\n\n---\n\n'

    html_body = markdown.markdown(combined_md, extensions=['tables', 'fenced_code', 'toc'])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DS Guardian - Manual de Usuario</title>
<style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #1e293b; padding: 40px; max-width: 900px; margin: 0 auto; }}
    h1, h2, h3 {{ color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; }}
    h1 {{ font-size: 24pt; color: #0284c7; border-bottom: 3px solid #0284c7; margin-bottom: 20px; }}
    h2 {{ font-size: 16pt; margin-top: 24px; }}
    code {{ background-color: #f1f5f9; color: #0f172a; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
    pre {{ background-color: #0f172a; color: #f8fafc; padding: 14px; border-radius: 6px; overflow-x: auto; }}
    pre code {{ background-color: transparent; color: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
    th {{ background-color: #e2e8f0; font-weight: bold; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    temp_html = '/tmp/temp_ds_guardian_manual.html'
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[+] Generating PDF at: {output_pdf}")
    subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', temp_html, '--outdir', docs_dir], check=True)
    
    generated_temp_pdf = '/tmp/temp_ds_guardian_manual.pdf'
    if os.path.exists(generated_temp_pdf):
        os.rename(generated_temp_pdf, output_pdf)

    if os.path.exists(output_pdf):
        print(f"✅ PDF Manual successfully generated: {output_pdf}")

if __name__ == '__main__':
    build_pdf_manual()

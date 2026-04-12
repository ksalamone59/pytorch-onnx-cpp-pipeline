import sys
import os
import subprocess
import re

pwd = os.getcwd()

if len(sys.argv) != 2:
    print("Usage: python fix.py </path/to/filename_without_extension>.\nFilename for gnuplot and tex must be the same basename.")
    sys.exit(1)

# Needs files to be in a certain naming scheme; could be room for future improvement
path = sys.argv[1]
basename = os.path.basename(path)
dirpath = os.path.dirname(path)
if dirpath:
    os.chdir(dirpath)
gnufile = f"{basename}.gnu"
texfile = f"{basename}.tex"

title = ""
labels = []

# This handles extracting the title and labels from the gnuplot file before processing the entire file
with open(gnufile, "r") as f:
    pattern = r"set\s+(?:\w*label|title)\s+['\"]([^'\"]+)['\"]"
    for line in f:
        match_title_label = re.search(pattern,line)
        if match_title_label:
            if "title" in line:
                title = match_title_label.group(1)
            else:
                labels.append(match_title_label.group(1))
            
subprocess.run(["gnuplot", gnufile], check=True)

with open(texfile, "r") as f:
    lines = f.readlines()

# Truncate the file and rewrite it with the necessary fixes
with open(texfile, "w") as f:
    lastLine = None
    definedColorDict = {}
    fixed_title = False 
    fixed_labels = [False]*len(labels)
    for line in lines:
        # Add xcolor package
        if r"\documentclass{minimal}" in line:
            f.write(line)
            f.write(r"\usepackage{xcolor}" + "\n")
            continue 
        # Make titles and labels bigger from pre processing
        if not fixed_title and title and title in line:
            line = line.replace(title, r'\LARGE ' + title)
            fixed_title = True
        elif any(label in line for label in labels): 
            for i, label in enumerate(labels):
                if not fixed_labels[i] and label and label in line:
                    line = line.replace(label, r'\Large ' + label)
                    fixed_labels[i] = True
                    break
        # Fixing incorrect text coloring
        if lastLine and r"\colorrgb" in lastLine and r"%%" in lastLine:
            col = "{" + lastLine.split("{")[-1].split("}")[0] + "}"
            # Add color to dictionary
            if col not in definedColorDict:
                definedColorDict[col] = f"mycolor{len(definedColorDict)}"
                f.write(r"\definecolor{" + definedColorDict[col] + r"}{rgb}{" + lastLine.split("{")[-1].split("}")[0] + "}\n")
            line = re.sub(r"\\strut\{\}(.+?)(\}\}+%)", r"\\textcolor{" + definedColorDict[col] + r"}{\\strut{}\1}\2", line)
        # Fixing incorrect text coloring (alternative case)
        elif re.search(r"\\strut\{\}([^}]*)", line) and re.search(r"\\textcolor", lastLine):
            col = lastLine.split(r"\textcolor{")[-1].split("}")[0]
            line = re.sub(r"(\\strut\{\}[^}]*)", r"\\textcolor{" + col + r"}{\1}", line)
        # Fix scientific notation
        elif re.search(r"(10\^\{[^}]+\})", line) and not re.search(r"\$", line):
            line = re.sub(r"(10\^\{[^}]+\})", r"$\1$", line)
        # Make tic numbers bigger
        elif re.search(r"\\put", line) and re.search(r"\\strut{}\$[-+]?(?:\d*\.\d+|\d+)\$", line):
            line = re.sub(r'(\\strut{})\$\s*([-+]?(?:\d*\.\d+|\d+))\$', r'\1\\Large$\2$', line)
        lastLine = line
        f.write(line)

subprocess.run(["pdflatex", texfile], check=True)

os.chdir(pwd)
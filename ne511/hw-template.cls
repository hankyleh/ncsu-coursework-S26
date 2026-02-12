% ~~~ identification ~~~~~~~~~
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template}[2025/08/25 MA580 Homework Problems]
% ~~~ initial code ~~~~~~~
\LoadClass[12pt]{article}
\usepackage{graphicx}
\usepackage[left=1in, right=1in, top=1in, bottom=1in]{geometry}
\usepackage{titlesec}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{enumerate}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\usepackage{amsthm}
\usepackage{subcaption}


\usepackage{minted}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\renewcommand\qedsymbol{QED}
\renewcommand\theoremautorefname{Lemma }

\setlength{\parindent}{0em}
\setlength{\parskip}{1ex}
% ~~~ declaration of options ~~~~~~~~~
\renewcommand\thesubsection{\thesection (\alph{subsection})}
\titleformat{\section}{\normalfont\Large\bfseries}{Question \thesection}{1ex}{}
\titleformat{\subsection}[runin]{\normalfont\large\bfseries}{\newline\newline\hspace{2em}\thesubsection}{1ex}{}[\setlength{\parindent}{2em}]




\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\trace}{\mathrm{trace}}
\newcommand{\mat}[1]{\mathbf{{#1}}}
\newcommand*{\defeq}{\stackrel{\text{def}}{=}}
\newcommand{\norm}[1]{\left\| {#1} \right\|}
\newcommand{\ip}[2]{\left\langle{#1}, {#2}\right\rangle}
\renewcommand{\vec}[1]{{\mathchoice
                     {\mbox{\boldmath$\displaystyle{#1}$}}
                     {\mbox{\boldmath$\textstyle{#1}$}}
                     {\mbox{\boldmath$\scriptstyle{#1}$}}
                     {\mbox{\boldmath$\scriptscriptstyle{#1}$}}}}
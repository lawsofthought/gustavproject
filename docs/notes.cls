\NeedsTeXFormat{LaTeX2e}[1994/06/01]
\ProvidesClass{foobar}[2010/02/05 foobar class, inspired (ahem) by Tufte-handout class v2.0.0]

% sfsidenotes option -- typesets sidenotes in sans serif typeface
\newif\if@foobar@sfsidenotes \@foobar@sfsidenotestrue
\DeclareOption{rmsidenotes}{\@foobar@sfsidenotesfalse}
%%
% symmetric option -- puts marginpar space to the outside edge of the page
%   Note: this forces twoside

\newif\if@foobar@symmetric\@foobar@symmetricfalse
\DeclareOption{symmetric}{
  \@foobar@symmetrictrue
  \PassOptionsToClass{twoside}{article}
}

% these are some fonts, purely for experimentation
\newcommand{\foobarsize}{\fontsize{8}{10}\selectfont}
\newcommand{\foobarIXXsize}{\fontsize{9}{10}\selectfont}

%%
% justified option -- uses fully justified text (flush left and flush
% right) instead of ragged right.

\newif\if@foobar@justified\@foobar@justifiedtrue
\DeclareOption{rag}{\@foobar@justifiedfalse}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions

\LoadClass{article}

% redefine normalsize

%%
% Detect whether we're in two-side mode or not.  (Used to set up running
% heads later.)

\newif\if@foobar@twoside\let\if@foobar@twoside\if@twoside


\RequirePackage{geometry}
%\geometry{a4paper,includemp,width=170mm,marginparsep=10mm,marginparwidth=50mm}
\geometry{a4paper,left=24.8mm,top=27.4mm,bottom=34.4mm,headsep=2\baselineskip,textwidth=122mm,marginparsep=8.2mm,marginparwidth=34.4mm,textheight=49\baselineskip,headheight=\baselineskip}
%%
% Separation marginpars by a line's worth of space.

\setlength\marginparpush{\baselineskip}

%%
% Font for margin items

\if@foobar@sfsidenotes
  \newcommand{\@foobar@marginfont}{\normalfont\scriptsize\sffamily}
\else
  \newcommand{\@foobar@marginfont}{\normalfont\scriptsize}
\fi

% \RaggedRight allows hyphenation

\RequirePackage{ragged2e}
\setlength{\RaggedRightRightskip}{\z@ plus 0.08\hsize}



%
% Define a subtitle variable and initialize it
\newcommand{\subtitle}[1]{\def\@subtitle{#1}}
\subtitle{\@title}


%%
% Set the justification baesed on the `justified' class option

\newcommand{\@foobar@justification}{%
  \if@foobar@justified
    \justifying
  \else
    \RaggedRight
  \fi
}
%%
% Turn off section numbering

\setcounter{secnumdepth}{-1}

%\renewenvironment{equation}%
%{\begin{equation*}}%
%{\end{equation*}}
%
\renewenvironment{equation}{\[}{\]}
%%
% Tighten up space between displays (e.g., a figure or table) and make symmetric

\setlength\abovedisplayskip{6pt plus 2pt minus 4pt}
\setlength\belowdisplayskip{6pt plus 2pt minus 4pt}

%%
% To implement full-width display environments

\RequirePackage[strict]{chngpage}

% Compute length used for full-width displays

\newlength{\@foobar@overhang}
\setlength{\@foobar@overhang}{\marginparwidth}
\addtolength{\@foobar@overhang}{\marginparsep}

%%
% Alter \maketitle from article.cls

\renewcommand\maketitle{\par
  \global\let\and\relax
  \global\let\thanks\footnote
  \begingroup
    \newpage
    \global\@topnum\z@
    \@maketitle
  \endgroup
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  %\global\let\@author\@empty
  %\global\let\@date\@empty
  %\global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
}
\def\@maketitle{%
  \newpage
  %\begin{adjustwidth}{}{-\@foobar@overhang}%
  \noindent\rmfamily\huge{\it{\@title}}\vskip0.5\baselineskip
  %\end{adjustwidth}
  \noindent\small{\sffamily{\@author}}\\
  \noindent{\small{\sffamily{\@date}}}\vskip2.0\baselineskip
  \thispagestyle{plain}
}

%%
% Abstract
%
\newcommand{\@foobar@abstractsize}{\fontsize{8}{10}\selectfont}
\renewenvironment{abstract}%
 {\begin{center}\begin{minipage}{90mm}\sffamily\@foobar@abstractsize}
%    \begin{scriptsize}}%
{  \end{minipage}\end{center}\vskip2.0\baselineskip}
%
%%
% Require paralist package for tighter lists

\RequirePackage{paralist}

% Add rightmargin to compactenum

\def\@compactenum@{%
  \expandafter\list\csname label\@enumctr\endcsname{%
    \usecounter{\@enumctr}%
    \rightmargin=2em% added this
    \parsep\plparsep
    \itemsep\plitemsep
    \topsep\pltopsep
    \partopsep\plpartopsep
    \def\makelabel##1{\hss\llap{##1}}}}

%%
% Improved letterspacing of small caps and all-caps text.
%
% First, try to use the `microtype' package, if it's available. 
% Failing that, try to use the `soul' package, if it's available.
% Failing that, well, I give up.

\RequirePackage{textcase} % provides \MakeTextUppercase and \MakeTextLowercase
\newcommand{\allcaps}[1]{\MakeTextUppercase{#1}}
\newcommand{\smallcaps}[1]{\textsc{\MakeTextLowercase{#1}}}
\newcommand{\smallcapsspacing}[1]{#1}

\IfFileExists{microtype.sty}{%
  \RequirePackage[final]{microtype}
  % Set up letterspacing (using microtype package) -- requires pdfTeX v1.40+
  \renewcommand{\smallcapsspacing}[1]{\textls[50]{##1}}
  \renewcommand{\allcaps}[1]{\textls[200]{\MakeTextUppercase{##1}}}
  \renewcommand{\smallcaps}[1]{\textsc{\MakeTextLowercase{##1}}}
}{% microtype failed, check for soul
  \IfFileExists{soul.sty}{%
    \RequirePackage{soul}
    \sodef\allcapsspacing{\upshape}{0.15em}{0.65em}{0.6em}
    \sodef\smallcapsspacing{\scshape}{0.075em}{0.5em}{0.6em}
    \renewcommand{\allcaps}[1]{\allcapsspacing{\MakeTextUppercase{##1}}}
    \renewcommand{\smallcaps}[1]{\textsc{\MakeTextLowercase{##1}}}
  }{}% neither microtype nor soul are installed... giving up.
}

\DeclareTextFontCommand{\textsmallcaps}{\scshape}
\renewcommand{\textsc}[1]{\textsmallcaps{\smallcapsspacing{#1}}}


%%
% An environment for paragraph-style section

\providecommand\newthought[1]{\vspace{1.8\baselineskip plus 3pt minus 2pt}%
                           {\noindent\textsc{#1}}}


%%
% Transform existing \footnotes into \sidenotes
% Sidenote: ``Where God meant footnotes to go.'' ---Tufte

\RequirePackage[side,multiple]{footmisc}
\renewcommand{\footnotelayout}{\@foobar@marginfont\@foobar@justification}

% Override footmisc's definition to set the sidenote marks (numbers) inside the
% sidenote's text block.
\long\def\@makefntext#1{\@textsuperscript{\@foobar@marginfont\tiny\@thefnmark}\,\footnotelayout#1}

% Set the in-text footnote mark in the same typeface as the body text itself.
\def\@makefnmark{\hbox{\@textsuperscript{\normalfont\scriptsize\@thefnmark}}}

\let\sidenote\footnote

%%
% Sidenote without the footnote mark

\providecommand\marginnote[1]%
  {\marginpar{\@foobar@marginfont\@foobar@justification\noindent #1}}

%
%%
% Make Tuftian-style section headings

% TODO: I'd prefer to use the 'titlesec' package for this formatting, but
%       I'll do it old-style for now.  --Kevin

\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\large\it}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\it}}

% Subsubsection and following section headings shouldn't be used.
% See Bringhurst's _The Elements of Typography_, section 4.2.2.
\renewcommand\subsubsection{%
  \PackageError{tufte-handout}{\noexpand\subsubsection is undefined by this class.%
    \MessageBreak See Robert Bringhurst's _The Elements of 
    \MessageBreak Typographic Style_, section 4.2.2.
    \MessageBreak \noexpand\subsubsection was used}
    {From Bringhurst's _The Elements of Typographic Style_, section 4.2.2: Use as 
    \MessageBreak many levels of headings as you need, no more and no fewer.  Also see the many 
    \MessageBreak related threads on Ask E.T. at http://www.edwardtufte.com/.}
}

\renewcommand\paragraph{%
  \PackageError{tufte-handout}{\noexpand\paragraph is undefined by this class.%
    \MessageBreak See Robert Bringhurst's _The Elements of 
    \MessageBreak Typographic Style_, section 4.2.2.
    \MessageBreak \noexpand\paragraph was used}
    {From Bringhurst's _The Elements of Typographic Style_, section 4.2.2: Use as 
    \MessageBreak many levels of headings as you need, no more and no fewer.  Also see the many 
    \MessageBreak related threads on Ask E.T. at http://www.edwardtufte.com/.}
}

\renewcommand\subparagraph{%
  \PackageError{tufte-handout}{\noexpand\subparagraph is undefined by this class.%
    \MessageBreak See Robert Bringhurst's _The Elements of 
    \MessageBreak Typographic Style_, section 4.2.2.
    \MessageBreak \noexpand\subparagraph was used}
    {From Bringhurst's _The Elements of Typographic Style_, section 4.2.2: Use as 
    \MessageBreak many levels of headings as you need, no more and no fewer.  Also see the many 
    \MessageBreak related threads on Ask E.T. at http://www.edwardtufte.com/.}
}



%%
% The placeins package provides the \FloatBarrier command.  This forces
% LaTeX to place all of the floats before proceeding.  We'll use this to
% keep the float (figure and table) numbers in sequence.
\RequirePackage{placeins}

%%
% Margin figure environment

\newsavebox{\@foobar@marginfigbox}
\newenvironment{marginfigure}[1]
    [-1.2ex]
    {\FloatBarrier% process all floats before this point so the figure numbers stay in order.
    \begin{lrbox}{\@foobar@marginfigbox}
    \begin{minipage}{\marginparwidth}
        \@foobar@marginfont
        %\def\@captype{figure}
        %\vspace*{#1}
        %\@foobar@justification
    }    
    {\end{minipage}%
     \end{lrbox}%
     \marginpar{\usebox{\@foobar@marginfigbox}}}


%%
% Margin table environment
\newsavebox{\@foobar@margintablebox}
\newenvironment{margintable}[1]
   [-1.2ex]
   {\FloatBarrier% process all floats before this point so the figure numbers stay in order.
   \begin{lrbox}{\@foobar@margintablebox}
   \begin{minipage}{\marginparwidth}
       \@foobar@marginfont
       \def\@captype{table}
       \vspace*{#1}
       \@foobar@justification
   }
   {\end{minipage}%
    \end{lrbox}%
    \marginpar{\usebox{\@foobar@margintablebox}}}


%%
% Full-width figure

\renewenvironment{figure*}[1]%
  [htbp]%
  {\@float{figure}[#1]%
   \if@foobar@symmetric
     \begin{adjustwidth}[]{}{-\@foobar@overhang}%
   \else
     \begin{adjustwidth}{}{-\@foobar@overhang}%
   \fi
   \begin{minipage}{\linewidth}}%
  {\end{minipage}%
   \end{adjustwidth}%
   \end@float}

%%
% Full-width table

\renewenvironment{table*}[1]
  [htbp]%
  {\@float{table}[#1]%
   \if@foobar@symmetric
     \begin{adjustwidth}[]{}{-\@foobar@overhang}%
   \else
     \begin{adjustwidth}{}{-\@foobar@overhang}%
   \fi
   \begin{minipage}{\linewidth}}%
  {\end{minipage}%
   \end{adjustwidth}%
   \end@float}

%%
% Full-page-width area

\newenvironment{fullwidth}
  {\if@foobar@symmetric
     \begin{adjustwidth}[]{}{-\@foobar@overhang}%
   \else
     \begin{adjustwidth}{}{-\@foobar@overhang}%
   \fi
  }
  {\end{adjustwidth}}

%%
% Format the captions in a style similar to the sidenotes

\RequirePackage[format=default,font={rm,scriptsize},justification=raggedright,singlelinecheck=false]{caption}

% if the `sfsidenotes' option is specified, set the captions in sf, too.
\if@foobar@sfsidenotes
  \captionsetup{font={sf,scriptsize}}
\else
  \captionsetup{font={rm,scriptsize}}
\fi

% if the `justified' option is specified, set the captions in flush left
% and flush right
\if@foobar@justified
  \captionsetup{justification=justified}
\else
  \captionsetup{justification=raggedright}
\fi

%%
% If the Palatino typeface (and its math symbol set) are installed, load them.

%\IfFileExists{palatino.sty}{%
%\IfFileExists{gfsartemisia-euler}{\usepackage[T1]{fontenc}
%\RequirePackage{gfsartemisia-euler}}{}
\usepackage[T1]{fontenc}
%\RequirePackage{gfsartemisia-euler}
\usepackage{mathpazo}
\linespread{1.05}
%  \RequirePackage{palatino}
%  \IfFileExists{mathpazo.sty}{\RequirePackage[osf,sc]{mathpazo}}{}
%}{} % if the Palatino typefaces aren't found, do nothing.
%

% Euler for math | Palatino for rm | Helvetica for ss | Courier for tt
%\renewcommand{\rmdefault}{ppl} % rm
%\linespread{1.05}        % Palatino needs more leading
\renewcommand{\sfdefault}{uop}
%\usepackage{courier} % tt
\usepackage{eulervm} % a better implementation of the euler package (not in gwTeX)
\normalfont






%%
% If the Bera Mono typeface is available, use it.

\IfFileExists{beramono.sty}{%
  \RequirePackage[T1]{fontenc}
  \RequirePackage[scaled=0.85]{beramono}
}{}


%%
% Sets up the running heads and folios.

\RequirePackage{fancyhdr}

% Set the header/footer width to be the body text block plus the margin
% note area.
\if@foobar@symmetric
  \fancyhfoffset[LE,RO]{\@foobar@overhang}
\else
  \fancyhfoffset[RE,RO]{\@foobar@overhang}
\fi

% The 'fancy' page style is the default style for all pages.
\fancyhf{} % clear header and footer fields
\if@foobar@twoside
  \fancyhead[LE]{\thepage\quad\smallcaps{\@author}}
  \fancyhead[RO]{\smallcaps{\@title}\quad\thepage}
\else
  \fancyhead[RE,RO]{{\it\@subtitle}\quad\thepage} % this is where we define header
\fi
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% The 'plain' page style is used on chapter opening pages.
\fancypagestyle{plain}{
  \fancyhf{} % clear header and footer fields
  \if@foobar@twoside
    \fancyfoot[LE,RO]{\thepage}
  \else
    \fancyfoot[RE,RO]{\thepage}
  \fi
}

% The 'empty' page style suppresses all headers and footers.
% It's used on title pages and `intentionally blank' pages.
\fancypagestyle{empty}{
  \fancyhf{} % clear header and footer fields
}

% Set the default page style to 'fancy'
\pagestyle{fancy}


%%
% Set raggedright and paragraph indentation for document

\AtBeginDocument{\@foobar@justification\setlength\parindent{1em}}


%%
% Prints the list of class options and their states.
\newcommand{\printclassoptions}{%
  \texttt{symmetric}---\if@foobar@symmetric true\else false\fi\\
  \texttt{a4paper}---\if@foobar@afourpaper true\else false\fi\\
  \texttt{twoside}---\if@foobar@twoside true\else false\fi\\
  \texttt{sfsidenotes}---\if@foobar@sfsidenotes true\else false\fi\\
  \texttt{justified}---\if@foobar@justified true\else false\fi
}

\endinput

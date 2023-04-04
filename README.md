# CA_FATA
The implementation of CA_FATA
# A toy example
\begin{tikzpicture}
\node[circle,draw,text = green] (a) at (7,1) {at1};
\node[circle,draw,text = red] (b) at (10,-2) {at2};
\node[circle,draw] (c) at (13,1) {at3};
\node[circle,draw,text=black,fill=lightgray] (d) at (10,0) {item};
\draw[green,->,thick] (a) to [in = 120, out  = -120] (d);
\node[text = green] at (8.5,0.90) {+0.52};
\draw[red,->,thick] (b) -- (d);
\node[text = red] at (10.4,-1) {-0.11};
\node at (11.5,0.7) {0};
\draw[->,thick] (c) to [in = 90, out  = -120] (d);
\end{tikzpicture} 

mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
\n\
[theme]\n\
base=\"dark\"\n\
primaryColor=\"#1bc4c5\"\n\
backgroundColor=\"#000000\"\n\
\n\
" > ~/.streamlit/config.toml


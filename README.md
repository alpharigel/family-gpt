# family-gpt
Personal GPT assistant for every member of your family, via Streamlit, Langchain and GPT-4


# Development

A few things to get you started locally:

1. git clone to your local filesystem
1. create a virtual environment locally
1. `pip install -r requirements`
1. Create a local postgres sql database to store your local configuration, or a dev database for deployment.
1. create a `.env` file with your environment variables:
```
OPENAI_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8080/  # this will be changed with your deployment
DATABASE_URL="dbname=postgres user=postgres  password=XXX" # or whatever you have for your local database
```
1. `streamlit run app.py `

If you dont want to go through the hassle of setting up google credentials or a postgres SQL Database right away
you can comment out that code and run the app without saving messages or user-specific authorization and configuration.

# Deployment

I've successfully deployed this app to render.com and digitalocean.com after just providing the github repo. 
To store the configuration and message history you need a postgres database in those environments. 

The google authentication for the app is necessary to track different users. Each user will start with 
the prompt in "base_prompt.py" and then can customize to their liking. 

To setup google credentials go to: https://console.cloud.google.com/apis/credentials/oauthclient 
You can use an existing or new project, but you will need a new "OAuth 2.0 Client ID" with the type
web application.

In the configuration for the google authentiatcation, make sure to put your localhost:port, as well
as any deployment locations in both "Authorized JavaScript origins" and "Authorized redirect URIs".
The "Authorized redirect URIs" are pretty touchy, so I suggest both the url with and with out a final "/"
in the configuration.

# Development Notes

A couple of things here

## Streamlit messages

I orginally used the pip package for messages, but needed markdown support and very much disliked the 
icons used for the bot and the human. Instead, I used my AI assistant to create markdown boxes with 
css formatting to look similar to iMessage or Teams chats. In the future if streamlit creates native
support, or there's a version that lets you skip the avatars, we can pull that in.

## Streamlit google authentication

This pulls directly from an open-source repo for the streamlit google authentication connection. This works,
but is a little less than ideal. In the future it would be great to have a streamlit component or pypi
repo to pull from here.

## Super user

Passing ?superuser=True at the end of your configuration will launch into super user mode. Right now there
are just additional prompt configuration that are hidden by default, and exposed to the super user mode. If
more superuser powers are added in the future, we should match the superuser user id against approved super
users in the app environment.

## Agent configuration

More improvements in the agent configuration are possible here. In particular, I'm working on integrating
zep (https://docs.getzep.com/) for long-term storage of messages. 


# Contributions Welcome!

Contributions are very much welcome. Please submit pull requests or issue tickets for anything you see.




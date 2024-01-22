FROM ubuntu:latest

# On install cargo, unzip et wget
RUN apt update \
 && apt install cargo -y \
 && apt install -y unzip \
 && apt install -y wget \
 && apt clean

# On créer un répertoire /app/moseiik
ADD . /app/moseiik

# On ajoute tous les fichiers du repo dans /app/moseiik
COPY . /app/moseiik

# On supprime le Dockerfile qui est initialement dans le repo (nous n'en avons pas besoin dans l'image elle même)
RUN rm /app/moseiik/Dockerfile

# On va dans /app/ et on télécharge nos tiles au format ZIP
WORKDIR /app/
RUN wget "https://filesender.renater.fr/download.php?token=178558c6-7155-4dca-9ecf-76cbebeb422e&files_ids=33679270" -O images.zip

# On unzip nos tiles et on les mets dans asserts
RUN unzip -o /app/images.zip -d /app/moseiik/assets/

WORKDIR /app/moseiik

# On build le projet
RUN cargo build --release

# Lance la commande: cargo test --release, a chaque fois que l'on run notre docker
ENTRYPOINT ["cargo", "test", "--release"]
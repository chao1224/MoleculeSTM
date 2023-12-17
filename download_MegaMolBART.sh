cd data
mkdir pretrained_MegaMolBART
cd pretrained_MegaMolBART

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1O5u8b_n93HOrsjN1aezq6NhZojh-6dEe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1O5u8b_n93HOrsjN1aezq6NhZojh-6dEe" -O models.zip && rm -rf /tmp/cookies.txt
unzip models.zip -d ./
rm -r ./__MACOSX
rm models.zip
mv models/megamolbart/* .
rm -rf models

cd ..

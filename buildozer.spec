[app]
title = SilenceAI
package.name = silenceai
package.domain = org.silenceai

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,vtt

version = 0.1

requirements = python3,kivy,opencv,mediapipe,pyttsx3,joblib,tensorflow,numpy,android

orientation = portrait
fullscreen = 1
android.permissions = CAMERA,INTERNET,RECORD_AUDIO,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

android.api = 31
android.minapi = 21
android.ndk = 23b
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1

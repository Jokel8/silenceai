<!DOCTYPE html>
<html lang="de">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>JKintelligence</title>
    <meta name="description" content="Die Projektseite eines Teams aus Informatikstudenten, die mit der Entwicklung fortschrittlicher KI-Lösungen komplexe Herausforderungen in verschiedenen Bereichen bewältigen möchte." />
    <link rel="stylesheet" href="style.css" />
    <meta name="google-site-verification" content="LGcRxz7SR5LqDE1bHsg2EjMXuIS09v5HgPuNlF5-V2o" />
    <script src="background.js" defer></script>
</head>

<body>
    <canvas id="nnCanvas"></canvas>

    <div class="section" id="cover-section">
        <div class="logo-box">
            <img src="img/jki_logo.png" alt="JKintelligence Logo"
                style="width: 150px; display: block; margin: 20px auto;" />
            JKintelligence
        </div>
    </div>

    <div class="section">
        <div class="text-box">
            <h2>Wer ist JKintelligence?</h2>
            <p>JKintelligence ist ein Team aus Informatikstudenten, das mit der Entwicklung fortschrittlicher KI-Lösungen komplexe Herausforderungen in verschiedenen Bereichen bewältigen möchte.<br><br> Dabei sind unsere Ziele:</p>
            <ul>
                <li>Einen positiven Beitrag zur Geselschaft zu leisten</li>
                <li>Nachhaltigkeit und das Bewusstsein dafür zu stärken</li>
                <li>Erweiterbarkeit, Anpassbarkeit und Quelloffenheit bei allen Projekten zu gewährleisten</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <div class="text-box">
            <h2>Silence AI</h2>
            <p>Das erste Projekt von JKintelligence ermöglicht eine barrierefreie Kommunikation für taubstumme Personen durch ein Zusammenspiel von 5 KI-Systemen:
            </p>
            <ul>
                <li>Preprocessing</li>
                <li>Keypoint-Erkennung</li>
                <li>Gestenerkennung</li>
                <li>Postprocessing</li>
                <li>Sprachsynthese</li>
            </ul>
            <p>Der Teil der Gestenerkennung hat dabei die höchste Komplexizität. Unser selbsttrainiertes neuronale Netz aus 4 Schichten errreicht dabei immerhin 21% Genauigkeit auf Trainingsdatensätze und durch ein aufwenidigeres Training erhoffen wir uns zeitnah bessere Ergebnisse</p>
            <p>Zudem versuchen wir eine Mobile App zu entwickeln, die es ermöglichen soll, direkt über das Handy Gebärdensprache zu übersetzen.</p>
        </div>
    </div>

    <div class="section">
        <div class="text-box">
            <h2>Weiteres Engagement:</h2>
            <p>Darüber hinaus enagieren wir uns für eine größere KI-Awareness an Schulen, da wir der Meinung sind, dass es für ausnahmslos jede Person in der Zukunft wichtig sein wird, zu wissen wie man KIs einsetzt, was dahinter steckt und wie man sie kritisch bewertet. Dabei haben wir bereits:</p>
            <ul>
                <li>Workshops für Lehrer an mehreren Schulen in Rheinland-Pfalz gehalten</li>
                <li>Bei der Jahrestagung der Landesdirektorenvereinigung der Gymnasien von Rheinland-Pfalz unsere Vision für die Zukunft vorgestelt</li>
            </ul>
        </div>
    </div>

    <button class="legal-button" id="legal-button"
        onclick="window.location.href='https://jonas.fam-klupsch.de/?p=Impressum'">
        Impressum
    </button>

    <script>
        if (window.location.search.includes('?internal')) {
            const button = document.getElementById('legal-button');
            if (button) {
                button.remove();
            }
        }
    </script>

    <?php
    //Aufruf erfassen
    try {
        $url = "https://jonas.fam-klupsch.de/jkservices.php?sitename=jkteam";
        $response = file_get_contents($url);
    } catch (Exception $e) {
    }
    ?>
</body>

</html>
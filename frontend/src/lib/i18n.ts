import i18n from "i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import { initReactI18next } from "react-i18next";

import commonPl from "@/locales/pl/common.json";
import sectionsPl from "@/locales/pl/sections.json";
import commonEn from "@/locales/en/common.json";
import sectionsEn from "@/locales/en/sections.json";

const resources = {
  pl: {
    common: commonPl,
    sections: sectionsPl,
  },
  en: {
    common: commonEn,
    sections: sectionsEn,
  },
} as const;

void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: "pl",
    supportedLngs: ["pl", "en"],
    defaultNS: "common",
    ns: ["common", "sections"],
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ["localStorage", "navigator", "htmlTag"],
      caches: ["localStorage"],
      lookupLocalStorage: "i18nextLng",
    },
    react: {
      useSuspense: false,
    },
  });

i18n.on("languageChanged", (lng) => {
  document.documentElement.lang = lng;
});

export default i18n;

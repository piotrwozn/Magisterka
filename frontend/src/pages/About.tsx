import { About as AboutSection } from "@/components/sections/About";
import { TechStack } from "@/components/sections/TechStack";
import { Timeline } from "@/components/sections/Timeline";

export function About() {
  return (
    <div className="pt-8">
      <AboutSection />
      <TechStack />
      <Timeline />
    </div>
  );
}

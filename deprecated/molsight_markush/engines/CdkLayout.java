import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.layout.StructureDiagramGenerator;
import org.openscience.cdk.io.MDLV2000Writer;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;

import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * CDK 2D-layout driver for render_via_engine.py (run as a single-file
 * source program, JEP 330):
 *
 *   java -cp &lt;cdk-bundle.jar&gt; CdkLayout.java &lt;smiles-file&gt; &lt;out-mol-file&gt;
 *
 * Reads one SMILES from the first line of &lt;smiles-file&gt;, lays it out with
 * the StructureDiagramGenerator and writes a V2000 MOL block carrying the
 * generated 2D coordinates.  Exit codes: 2 usage, 1 layout failure
 * (diagnostic prefixed CDK_INVALID_SMILES / CDK_LAYOUT_FAILED on stderr).
 */
public final class CdkLayout {

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("usage: CdkLayout <smiles-file> <out-mol-file>");
            System.exit(2);
            return;
        }
        String smiles = new String(Files.readAllBytes(Paths.get(args[0])),
                "UTF-8").trim();
        if (smiles.isEmpty()) {
            System.err.println("CDK_INVALID_SMILES: empty smiles input");
            System.exit(1);
            return;
        }
        SmilesParser parser = new SmilesParser(
                SilentChemObjectBuilder.getInstance());
        IAtomContainer mol;
        try {
            mol = parser.parseSmiles(smiles);
        } catch (InvalidSmilesException e) {
            System.err.println("CDK_INVALID_SMILES: " + e.getMessage());
            System.exit(1);
            return;
        }
        StructureDiagramGenerator sdg = new StructureDiagramGenerator();
        sdg.setMolecule(mol, true);
        try {
            sdg.generateCoordinates();
        } catch (Exception e) {
            System.err.println("CDK_LAYOUT_FAILED: " + e);
            System.exit(1);
            return;
        }
        IAtomContainer laid = sdg.getMolecule();
        try (FileWriter out = new FileWriter(args[1]);
             MDLV2000Writer writer = new MDLV2000Writer(out)) {
            writer.write(laid);
        }
        System.err.println("CDK_LAYOUT_OK atoms=" + laid.getAtomCount()
                + " bonds=" + laid.getBondCount());
    }
}
